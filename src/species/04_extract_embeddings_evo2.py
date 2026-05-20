"""
04_extract_embeddings_evo2.py
-----------------------------
Loads Evo 2 7B in fp16, freezes all weights, and extracts per-layer
mean-pooled embeddings for every fragment in fragments.tsv.

Memory budget (46 GB GPU)
--------------------------
  Evo 2 7B weights in fp16   ≈ 14 GB
  Activations (1 fwd pass)   ≈  4–8 GB (depends on seq len)
  Remaining headroom          ≈ 24–28 GB   → safe for 1024 bp fragments
  
  DO NOT increase frag_len beyond 4096 without checking memory first.
  Use --batch_size 1 if you get OOMs; that is the safe default.

Evo 2 HuggingFace model ID: "arcinstitute/evo-2-7b"
  → loads with AutoModelForCausalLM + trust_remote_code=True

Outputs
-------
data/embeddings/evo2/
    layer_{i:02d}.h5    — HDF5, shape (N_frags, hidden_dim), float32
    frag_ids.txt        — fragment IDs in the same row order as the HDF5

Usage
-----
  python 04_extract_embeddings_evo2.py \
      --fragments data/fragments/fragments.tsv \
      --out_dir data/embeddings/evo2 \
      --batch_size 1 \
      --frag_len 1024 \
      --device cuda
"""

import argparse
import gc
import os
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
    sequences:  list[str],
    tokenizer,
    model,
    n_layers:   int,
    device:     str,
    frag_len:   int,
) -> list[np.ndarray]:
    """
    Given a list of DNA sequences, returns a list of arrays,
    one per layer, each of shape (len(sequences), hidden_dim).

    Mean-pools over the sequence (token) dimension per layer.
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

    # hidden_states: tuple of (batch, seq_len, hidden_dim), one per layer
    # Index 0 is the embedding layer; 1..n_layers are transformer layers.
    hidden_states = outputs.hidden_states   # tuple len = n_layers + 1

    per_layer: list[np.ndarray] = []
    mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)

    for layer_idx, hs in enumerate(hidden_states):
        # hs: (B, T, D) in fp16
        hs_f32 = hs.float()
        # Masked mean pool
        summed = (hs_f32 * mask_expanded).sum(dim=1)         # (B, D)
        lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)   # (B, 1)
        pooled  = (summed / lengths).cpu().numpy()            # (B, D)
        per_layer.append(pooled.astype(np.float32))

    return per_layer   # list[n_layers+1] of (B, D) float32 arrays


# ---------------------------------------------------------------------------
# HDF5 writer helpers
# ---------------------------------------------------------------------------

def init_h5_files(out_dir: Path, n_layers: int, n_frags: int, hidden_dim: int):
    """Creates one HDF5 file per layer, pre-allocated."""
    handles = {}
    for layer_idx in range(n_layers + 1):
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragments",   default="data/fragments/fragments.tsv")
    parser.add_argument("--out_dir",     default="data/embeddings/evo2")
    parser.add_argument("--batch_size",  type=int, default=1,
                        help="Keep at 1 for 46 GB GPU with 1 kb fragments. "
                             "Increase to 2–4 only if memory allows.")
    parser.add_argument("--frag_len",    type=int, default=1024)
    parser.add_argument("--device",      default="cuda")
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
    n_outputs = n_layers + 1   # includes embedding layer

    # --- Write frag_ids so row order is always recoverable ---
    id_path = out_dir / "frag_ids.txt"
    with open(id_path, "w") as f:
        f.write("\n".join(frag_ids))

    # --- Determine resume offset ---
    start_row = 0
    if args.resume and (out_dir / "layer_00.h5").exists():
        with h5py.File(out_dir / "layer_00.h5", "r") as fh:
            # Check how many rows are non-zero as a proxy for written rows
            # (a more robust approach would store a cursor file)
            cursor_path = out_dir / ".cursor"
            if cursor_path.exists():
                start_row = int(cursor_path.read_text().strip())
                print(f"Resuming from row {start_row}")

    # --- Init or reopen HDF5 files ---
    h5_handles = init_h5_files(out_dir, n_layers, n_frags, hidden_dim)

    # --- Batch forward passes ---
    batch_size = args.batch_size
    print(f"\nExtracting embeddings (batch_size={batch_size}) ...")

    for row_start in range(start_row, n_frags, batch_size):
        row_end   = min(row_start + batch_size, n_frags)
        batch_seq = sequences[row_start:row_end]

        try:
            per_layer = extract_layer_embeddings(
                batch_seq, tokenizer, model, n_layers, args.device, args.frag_len
            )
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at row {row_start}. Try --batch_size 1 or shorter --frag_len.")
            raise

        for layer_idx, layer_embs in enumerate(per_layer):
            h5_handles[layer_idx]["embeddings"][row_start:row_end] = layer_embs

        # Update cursor
        (out_dir / ".cursor").write_text(str(row_end))

        if row_start % 100 == 0 or row_end == n_frags:
            pct = 100 * row_end / n_frags
            print(f"  {row_end}/{n_frags} ({pct:.1f}%)  ", end="\r")

        # Explicit memory management to avoid CUDA fragmentation
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nDone. Embeddings saved to {out_dir}")
    close_h5_files(h5_handles)

    # Brief sanity check
    with h5py.File(out_dir / "layer_00.h5", "r") as fh:
        print(f"Embedding layer shape: {fh['embeddings'].shape}")
    with h5py.File(out_dir / f"layer_{n_layers:02d}.h5", "r") as fh:
        print(f"Final layer shape:     {fh['embeddings'].shape}")


if __name__ == "__main__":
    main()
