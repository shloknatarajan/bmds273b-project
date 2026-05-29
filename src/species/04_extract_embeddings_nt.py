"""
04_extract_embeddings_nt_v2.py
-----------------------------------
Extracts per-layer mean-pooled embeddings from Nucleotide Transformer v2 2.5B
(InstaDeepAI/nucleotide-transformer-2.5b-multi-species) using HuggingFace
transformers with output_hidden_states=True.

NT v2 uses AutoModelForMaskedLM and returns hidden states as a tuple of
(n_layers + 1) tensors, each shaped (batch, seq_len, hidden_dim). Index 0 is
the embedding layer output; indices 1..N are transformer block outputs.
The model has 33 transformer layers (indices 1..33 in hidden_states).

Layer addressing (--layers argument)
--------------------------------------
  "layer.N"   N = 0..33   (0 = embedding layer, 1..33 = transformer blocks)
              "layer.0"   → hidden_states[0]   (token embedding output)
              "layer.33"  → hidden_states[33]  (last transformer block)

Outputs
-------
processed_data/embeddings/nt_v2/
    <safe_layer_name>.h5   one HDF5 per requested layer
    frag_ids.txt           fragment IDs in row order
"""

import argparse
import gc
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


MODEL_NAME = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"

# NT v2 2.5B has 33 transformer blocks + 1 embedding layer = 34 hidden states
NT_NUM_LAYERS = 33  # transformer blocks; total hidden_states = 34


def load_nt_v2(device: str):
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    print("  Model loaded.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Layer name parsing
# ---------------------------------------------------------------------------

def parse_layer_index(layer_name: str) -> int:
    """
    Converts 'layer.N' → integer index N for hidden_states tuple.
    Valid range: 0 .. NT_NUM_LAYERS (i.e. 0..33).
    """
    parts = layer_name.split(".")
    if len(parts) != 2 or parts[0] != "layer":
        raise ValueError(
            f"Invalid layer name '{layer_name}'. "
            f"Expected 'layer.N' where N is 0..{NT_NUM_LAYERS}."
        )
    idx = int(parts[1])
    if not (0 <= idx <= NT_NUM_LAYERS):
        raise ValueError(
            f"Layer index {idx} out of range. Valid: 0..{NT_NUM_LAYERS}."
        )
    return idx


# ---------------------------------------------------------------------------
# Per-sequence embedding extraction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_layer_embeddings(
    sequence: str,
    tokenizer,
    model,
    layer_names: list,
    device: str,
    model_max_length: int,
) -> dict:
    """
    Returns {layer_name: (hidden_dim,) float32 numpy array}.

    NT v2 returns hidden_states as a tuple of tensors shaped
    (1, seq_len, hidden_dim). We mean-pool over real (non-padding) tokens.
    """
    tokens = tokenizer(
        sequence,
        return_tensors="pt",
        padding="max_length",
        max_length=model_max_length,
        truncation=True,
    )
    input_ids      = tokens["input_ids"].to(device)       # (1, seq_len)
    attention_mask = (input_ids != tokenizer.pad_token_id) # (1, seq_len) bool

    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # hidden_states: tuple of (1, seq_len, hidden_dim) tensors, length = 34
    hidden_states = outputs["hidden_states"]

    # Mask for mean-pooling: (1, seq_len, 1) float
    mask_fp = attention_mask.float().unsqueeze(-1).cpu()  # (1, seq_len, 1)
    n_real  = mask_fp.sum(dim=1, keepdim=True)            # (1, 1, 1)

    out: dict = {}
    for name in layer_names:
        idx    = parse_layer_index(name)
        hidden = hidden_states[idx].float().cpu()          # (1, seq_len, hidden_dim)
        # Mean pool over real tokens
        pooled = (hidden * mask_fp).sum(dim=1) / n_real.squeeze(-1)  # (1, hidden_dim)
        out[name] = pooled.squeeze(0).numpy().astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def safe_filename(layer_name: str) -> str:
    return layer_name.replace(".", "_").replace("/", "_")


def init_h5_files(out_dir: Path, layer_names: list, n_frags: int, hidden_dim: int):
    handles = {}
    for name in layer_names:
        path = out_dir / f"{safe_filename(name)}.h5"
        if path.exists():
            path.unlink()
        fh = h5py.File(path, "w")
        fh.create_dataset(
            "embeddings",
            shape=(n_frags, hidden_dim),
            dtype="float32",
            chunks=(min(256, n_frags), hidden_dim),
            compression="gzip",
            compression_opts=4,
        )
        fh.attrs["layer_name"] = name
        handles[name] = fh
    return handles


def close_h5_files(handles: dict):
    for fh in handles.values():
        fh.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragments",        default="processed_data/fragments.tsv")
    parser.add_argument("--out_dir",          default="processed_data/embeddings/nt_v2")
    parser.add_argument("--device",           default="cuda:0")
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=1000,
        help=(
            "Padded token length passed to the tokenizer. NT v2 max is "
            "tokenizer.model_max_length (typically 2048). "
            "1000 tokens ≈ 6000 bp; lower values reduce VRAM and runtime."
        ),
    )
    parser.add_argument(
        "--layers",
        default="layer.1,layer.12,layer.24,layer.32",
        help=(
            "Comma-separated layer names. "
            "'layer.0' = embedding output; "
            "'layer.1'..'layer.33' = transformer block outputs. "
            "layer.33 is equivalent to the last hidden state used in the "
            "HuggingFace model card example (hidden_states[-1])."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_names = [s.strip() for s in args.layers.split(",") if s.strip()]
    if not layer_names:
        raise ValueError("--layers is empty.")

    # Validate layer names up front
    for name in layer_names:
        parse_layer_index(name)
    print(f"Layers: {layer_names}")

    # --- Load fragments ---
    print("Loading fragments ...")
    df        = pd.read_csv(args.fragments, sep="\t")
    sequences = df["seq"].tolist()
    frag_ids  = df["frag_id"].tolist()
    n_frags   = len(df)
    print(f"  {n_frags} fragments")

    # --- Load model ---
    tokenizer, model = load_nt_v2(args.device)

    # Override max_length if user exceeds model's own limit
    model_max = tokenizer.model_max_length
    if args.model_max_length > model_max:
        print(
            f"  WARNING: --model_max_length {args.model_max_length} > "
            f"tokenizer.model_max_length {model_max}. Clamping to {model_max}."
        )
        args.model_max_length = model_max
    print(f"  Effective max_length: {args.model_max_length}")

    # --- Probe hidden_dim ---
    print("Probing hidden dimension ...")
    probe = extract_layer_embeddings(
        "ACGTACGT", tokenizer, model, layer_names,
        args.device, args.model_max_length,
    )
    for name, v in probe.items():
        print(f"  {name}: {v.shape}")
    hidden_dim = next(iter(probe.values())).shape[-1]
    print(f"  hidden_dim = {hidden_dim}")

    # --- Write frag_ids ---
    (out_dir / "frag_ids.txt").write_text("\n".join(frag_ids))

    # --- Resume ---
    start_row = 0
    if args.resume:
        cursor_path = out_dir / ".cursor"
        if cursor_path.exists():
            start_row = int(cursor_path.read_text().strip())
            print(f"Resuming from row {start_row}")

    # --- Init HDF5 ---
    if start_row == 0:
        h5_handles = init_h5_files(out_dir, layer_names, n_frags, hidden_dim)
    else:
        h5_handles = {
            name: h5py.File(out_dir / f"{safe_filename(name)}.h5", "r+")
            for name in layer_names
        }

    # --- Extract ---
    print("\nExtracting embeddings ...")
    for i in range(start_row, n_frags):
        try:
            per_layer = extract_layer_embeddings(
                sequences[i], tokenizer, model, layer_names,
                args.device, args.model_max_length,
            )
        except torch.cuda.OutOfMemoryError:
            print(
                f"\nOOM at fragment {i}. "
                f"Try a smaller --model_max_length (currently {args.model_max_length})."
            )
            raise

        for name, vec in per_layer.items():
            h5_handles[name]["embeddings"][i] = vec

        (out_dir / ".cursor").write_text(str(i + 1))

        if i % 100 == 0 or i == n_frags - 1:
            print(f"  {i + 1}/{n_frags} ({100*(i+1)/n_frags:.1f}%)  ", end="\r")

        if i % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nDone. Embeddings saved to {out_dir}")
    close_h5_files(h5_handles)

    first = safe_filename(layer_names[0])
    with h5py.File(out_dir / f"{first}.h5", "r") as fh:
        print(f"{first}.h5  shape: {fh['embeddings'].shape}")


if __name__ == "__main__":
    main()
