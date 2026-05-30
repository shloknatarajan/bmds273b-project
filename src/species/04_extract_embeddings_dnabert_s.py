"""
04_extract_embeddings_dnabert_s.py
-----------------------------------
Extracts per-layer embeddings from DNABERT-S (zhihan1996/DNABERT-S) using
HuggingFace transformers with forward hooks.

DNABERT-S uses a custom BertLayer that drops the batch dimension, returning
(seq_len, hidden_dim) instead of the standard (batch, seq_len, hidden_dim).
The pooler still returns (1, hidden_dim).

Layer addressing
----------------
"encoder.layer.N"  N = 0..11
"pooler"           [CLS] pooler output

Outputs
-------
processed_data/embeddings/dnabert_s/
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
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "zhihan1996/DNABERT-S"


def load_dnabert_s(device: str):
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    print("  Model loaded.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Hook-based layer capture
# ---------------------------------------------------------------------------

class LayerCaptureHooks:
    """Stores raw hook outputs; all shape handling is done after pop()."""

    def __init__(self, model, layer_names: list):
        self._storage: dict = {}
        self._handles = []
        for name in layer_names:
            module = self._resolve(model, name)
            self._handles.append(module.register_forward_hook(self._hook(name)))

    @staticmethod
    def _resolve(model, name: str):
        m = model
        for part in name.split("."):
            m = m[int(part)] if part.isdigit() else getattr(m, part)
        return m

    def _hook(self, name: str):
        def fn(module, input, output):
            # Store raw output on CPU immediately; unwrap tuples here so
            # pop() always returns plain tensors.
            t = output[0] if isinstance(output, (tuple, list)) else output
            self._storage[name] = t.detach().float().cpu()
        return fn

    def pop(self) -> dict:
        out = dict(self._storage)
        self._storage.clear()
        return out

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Per-sequence embedding extraction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_layer_embeddings(
    sequence: str,
    tokenizer,
    model,
    hooks: LayerCaptureHooks,
    device: str,
    model_max_length: int = 512,
) -> dict:
    """
    Returns {layer_name: (hidden_dim,) float32 numpy array}.

    DNABERT-S BertLayer returns (seq_len, hidden_dim) — no batch dim.
    The pooler returns (1, hidden_dim).
    We detect which case we have by shape and pool accordingly.
    """
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        max_length=model_max_length,
        truncation=True,
        padding=False,
    )
    input_ids      = inputs["input_ids"].to(device)       # (1, seq_len)
    attention_mask = inputs["attention_mask"]              # (1, seq_len) on CPU

    model(input_ids=input_ids, attention_mask=attention_mask.to(device))

    captured = hooks.pop()   # {name: tensor} all on CPU

    # Build a 1-D boolean mask over real tokens: shape (seq_len,)
    token_mask = attention_mask.squeeze(0).bool()         # (seq_len,)

    out: dict = {}
    for name, hidden in captured.items():
        # hidden is (seq_len, hidden_dim)  for encoder layers  — dim()==2, shape[0]>1
        #        or (1, hidden_dim)        for the pooler      — dim()==2, shape[0]==1
        if hidden.dim() == 2 and hidden.shape[0] > 1:
            # Encoder layer: mean-pool over real (non-padding) tokens
            real_tokens = hidden[token_mask[:hidden.shape[0]]]  # (n_real, hidden_dim)
            vec = real_tokens.mean(dim=0).numpy().astype(np.float32)
        elif hidden.dim() == 2 and hidden.shape[0] == 1:
            # Pooler: already a single vector
            vec = hidden.squeeze(0).numpy().astype(np.float32)
        elif hidden.dim() == 3:
            # Standard HF shape (1, seq_len, hidden_dim) — just in case
            mask_3d = attention_mask.float().unsqueeze(-1)  # (1, seq_len, 1)
            pooled = (hidden * mask_3d).sum(dim=1) / mask_3d.sum(dim=1)
            vec = pooled.squeeze(0).numpy().astype(np.float32)
        else:
            raise RuntimeError(
                f"Unexpected hidden shape for layer '{name}': {tuple(hidden.shape)}"
            )
        out[name] = vec

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
    parser.add_argument("--out_dir",          default="processed_data/embeddings/dnabert_s")
    parser.add_argument("--device",           default="cuda:0")
    parser.add_argument("--model_max_length", type=int, default=512,
                        help="Max subword tokens. DNABERT-S supports up to 2000; "
                             "512 is safe for most GPUs.")
    parser.add_argument(
        "--layers",
        default=(
            "encoder.layer.0,"
            "encoder.layer.5,"
            "encoder.layer.9,"
            "encoder.layer.10,"
            "encoder.layer.11,"
            "pooler"
        ),
        help="Comma-separated layer names. encoder.layer.N (N=0..11) or 'pooler'.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_names = [s.strip() for s in args.layers.split(",") if s.strip()]
    if not layer_names:
        raise ValueError("--layers is empty.")
    print(f"Layers: {layer_names}")

    # --- Load fragments ---
    print("Loading fragments ...")
    df = pd.read_csv(args.fragments, sep="\t")
    sequences = df["seq"].tolist()
    frag_ids  = df["frag_id"].tolist()
    n_frags   = len(df)
    print(f"  {n_frags} fragments")

    # --- Load model + hooks ---
    tokenizer, model = load_dnabert_s(args.device)
    hooks = LayerCaptureHooks(model, layer_names)
    print(f"  Hooks registered on: {layer_names}")

    # --- Probe hidden_dim ---
    print("Probing hidden dimension ...")
    probe = extract_layer_embeddings(
        "ACGT", tokenizer, model, hooks, args.device, args.model_max_length,
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
                sequences[i], tokenizer, model, hooks,
                args.device, args.model_max_length,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at fragment {i}. Try --model_max_length with a smaller value.")
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
    hooks.remove()
    close_h5_files(h5_handles)

    first = safe_filename(layer_names[0])
    with h5py.File(out_dir / f"{first}.h5", "r") as fh:
        print(f"{first}.h5 shape: {fh['embeddings'].shape}")


if __name__ == "__main__":
    main()
