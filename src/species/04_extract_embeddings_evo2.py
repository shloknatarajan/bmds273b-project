"""
04_extract_embeddings_evo2.py
-----------------------------
Loads Evo 2 7B in fp16, freezes all weights, and extracts mean-pooled
embeddings for every fragment in fragments.tsv.

By default this script saves **every** hidden state (embedding layer +
all transformer blocks) so that downstream probes can compare layers.
The Evo 2 paper highlights **layer 26** as the most informative single
layer, but we want the option to look at the others too. Pass
``--layers 26`` (or any comma-separated list, e.g. ``0,16,26,32``) to
restrict the set of layers written to disk.

Memory budget (46 GB GPU)
--------------------------
  Evo 2 7B weights in fp16   ~ 14 GB
  Activations (1 fwd pass)   ~  4-8 GB (depends on seq len)
  Remaining headroom         ~ 24-28 GB   -> safe for 1024 bp fragments

  DO NOT increase frag_len beyond 4096 without checking memory first.
  Use --batch_size 1 if you get OOMs; that is the safe default.

Evo 2 HuggingFace model ID: "arcinstitute/evo-2-7b"
  -> loads with AutoModelForCausalLM + trust_remote_code=True

Outputs
-------
processed_data/embeddings/evo2/
    layer_{i:02d}.h5    — HDF5, shape (N_frags, hidden_dim), float32
                          (one file per requested layer)
    frag_ids.txt        — fragment IDs in the same row order as the HDF5

Usage
-----
  # Default: extract every layer
  python 04_extract_embeddings_evo2.py

  # Restrict to the layer the Evo 2 paper highlights
  python 04_extract_embeddings_evo2.py --layers 26

  # Or a curated set
  python 04_extract_embeddings_evo2.py --layers 0,16,26,32
"""

import argparse
import gc
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Tokenizer + model loading
# ---------------------------------------------------------------------------

MODEL_ID = "arcinstitute/evo-2-7b"


def load_evo2(device: str):
    """
    Loads Evo 2 7B in fp16 with all weights frozen.
    output_hidden_states=True ensures all intermediate layer outputs
    are returned in the forward pass.
    """
    print(f"Loading tokenizer from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    print(f"Loading model in fp16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,          # "cuda" or "auto" for multi-GPU
        output_hidden_states=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Layers: {n_layers}, hidden_dim: {hidden_dim}")
    return tokenizer, model, n_layers, hidden_dim


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_layer_embeddings(
    sequences:    list[str],
    tokenizer,
    model,
    layers_keep:  list[int],
    device:       str,
    frag_len:     int,
) -> dict[int, np.ndarray]:
    """
    Given a list of DNA sequences, returns a dict mapping layer index
    to an array of shape (len(sequences), hidden_dim).

    Mean-pools over the sequence (token) dimension per layer.
    Only layers in ``layers_keep`` are returned to save memory.
    """
    encodings = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=frag_len + 16,   # small buffer for special tokens
    )
    input_ids      = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # hidden_states: tuple of (batch, seq_len, hidden_dim), one per layer.
    # Index 0 is the embedding layer; 1..n_layers are transformer layers.
    hidden_states = outputs.hidden_states

    per_layer: dict[int, np.ndarray] = {}
    mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
    lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)    # (B, 1)

    for layer_idx in layers_keep:
        hs = hidden_states[layer_idx].float()             # (B, T, D)
        summed = (hs * mask_expanded).sum(dim=1)          # (B, D)
        pooled = (summed / lengths).cpu().numpy()         # (B, D)
        per_layer[layer_idx] = pooled.astype(np.float32)

    return per_layer


# ---------------------------------------------------------------------------
# HDF5 writer helpers
# ---------------------------------------------------------------------------

def init_h5_files(out_dir: Path, layers: list[int], n_frags: int, hidden_dim: int):
    """Creates one HDF5 file per requested layer, pre-allocated."""
    handles = {}
    for layer_idx in layers:
        path = out_dir / f"layer_{layer_idx:02d}.h5"
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
        handles[layer_idx] = fh
    return handles


def close_h5_files(handles: dict):
    for fh in handles.values():
        fh.close()


def parse_layers_arg(layers_arg: str, n_layers: int) -> list[int]:
    """
    Resolve the --layers CLI argument into an explicit list of indices.
    ``n_layers`` is the number of transformer blocks; the hidden_states
    tuple has ``n_layers + 1`` entries (embedding layer at index 0).
    """
    if layers_arg.strip().lower() == "all":
        return list(range(n_layers + 1))
    out: list[int] = []
    for token in layers_arg.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0 or idx > n_layers:
            raise ValueError(
                f"Requested layer {idx} is out of range [0, {n_layers}]."
            )
        out.append(idx)
    if not out:
        raise ValueError("--layers parsed to an empty list.")
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragments",   default="processed_data/fragments.tsv")
    parser.add_argument("--out_dir",     default="processed_data/embeddings/evo2")
    parser.add_argument("--batch_size",  type=int, default=1,
                        help="Keep at 1 for 46 GB GPU with 1 kb fragments. "
                             "Increase to 2-4 only if memory allows.")
    parser.add_argument("--frag_len",    type=int, default=1024)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--layers",      default="all",
                        help="Comma-separated layer indices to save, or "
                             "'all' (default). The Evo 2 paper highlights "
                             "layer 26 as the most informative single layer.")
    parser.add_argument("--resume",      action="store_true",
                        help="If set, skip fragments already written (resume from crash).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load fragments ---
    print("Loading fragments ...")
    df = pd.read_csv(args.fragments, sep="\t")
    sequences = df["seq"].tolist()
    frag_ids  = df["frag_id"].tolist()
    n_frags   = len(df)
    print(f"  {n_frags} fragments to embed")

    # --- Load model ---
    tokenizer, model, n_layers, hidden_dim = load_evo2(args.device)

    layers_keep = parse_layers_arg(args.layers, n_layers)
    print(f"Saving embeddings for layers: {layers_keep}")

    # --- Write frag_ids so row order is always recoverable ---
    id_path = out_dir / "frag_ids.txt"
    with open(id_path, "w") as f:
        f.write("\n".join(frag_ids))

    # --- Determine resume offset ---
    start_row = 0
    if args.resume:
        cursor_path = out_dir / ".cursor"
        if cursor_path.exists():
            start_row = int(cursor_path.read_text().strip())
            print(f"Resuming from row {start_row}")

    # --- Init or reopen HDF5 files ---
    h5_handles = init_h5_files(out_dir, layers_keep, n_frags, hidden_dim)

    # --- Batch forward passes ---
    batch_size = args.batch_size
    print(f"\nExtracting embeddings (batch_size={batch_size}) ...")

    for row_start in range(start_row, n_frags, batch_size):
        row_end   = min(row_start + batch_size, n_frags)
        batch_seq = sequences[row_start:row_end]

        try:
            per_layer = extract_layer_embeddings(
                batch_seq, tokenizer, model, layers_keep,
                args.device, args.frag_len,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at row {row_start}. Try --batch_size 1 or shorter --frag_len.")
            raise

        for layer_idx, layer_embs in per_layer.items():
            h5_handles[layer_idx]["embeddings"][row_start:row_end] = layer_embs

        (out_dir / ".cursor").write_text(str(row_end))

        if row_start % 100 == 0 or row_end == n_frags:
            pct = 100 * row_end / n_frags
            print(f"  {row_end}/{n_frags} ({pct:.1f}%)  ", end="\r")

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nDone. Embeddings saved to {out_dir}")
    close_h5_files(h5_handles)

    # Brief sanity check on the first saved layer
    first_layer = layers_keep[0]
    with h5py.File(out_dir / f"layer_{first_layer:02d}.h5", "r") as fh:
        print(f"layer_{first_layer:02d}.h5 shape: {fh['embeddings'].shape}")


if __name__ == "__main__":
    main()
