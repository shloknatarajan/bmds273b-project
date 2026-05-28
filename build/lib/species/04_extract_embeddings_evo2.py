"""
04_extract_embeddings_evo2.py
-----------------------------
Loads Evo 2 7B via Arc Institute's `evo2` package (NOT HuggingFace
transformers — Evo 2 uses StripedHyena 2 and is loaded via Vortex).

The `evo2` package downloads the checkpoint from `arcinstitute/evo2_7b`
on first use; you don't need to authenticate.

Memory budget (46 GB GPU)
--------------------------
  Evo 2 7B weights in bf16   ~ 14 GB
  Activations (1 fwd pass)   ~  4-8 GB (depends on seq len)
  Remaining headroom         ~ 24-28 GB   -> safe for 1024 bp fragments

Layer addressing
----------------
Layers are named, not indexed. Examples:
  blocks.0.mlp.l3, blocks.15.mlp.l3, blocks.28.mlp.l3
The 7B has 32 blocks (0..31). Pass a comma-separated list to --layers.

Outputs
-------
processed_data/embeddings/evo2/
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

from evo2 import Evo2


MODEL_NAME = "evo2_7b"  # use "evo2_7b_base" for the 8K-context base model


def load_evo2():
    print(f"Loading Evo 2 ({MODEL_NAME}) — first run will download ~14 GB ...")
    evo2_model = Evo2(MODEL_NAME)
    evo2_model.model.eval()
    for p in evo2_model.model.parameters():
        p.requires_grad = False
    return evo2_model


@torch.inference_mode()
def extract_layer_embeddings(
    sequence: str,
    evo2_model,
    layer_names: list[str],
    device: str,
) -> dict[str, np.ndarray]:
    """
    One sequence in, dict {layer_name: (hidden_dim,) float32} out.
    Mean-pools over the token dimension.
    """
    input_ids = torch.tensor(
        evo2_model.tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    _, embeddings = evo2_model(
        input_ids,
        return_embeddings=True,
        layer_names=layer_names,
    )

    out: dict[str, np.ndarray] = {}
    for name, hs in embeddings.items():
        # hs shape: (1, seq_len, hidden_dim)
        pooled = hs.float().mean(dim=1).squeeze(0).cpu().numpy()
        out[name] = pooled.astype(np.float32)
    return out


def safe_filename(layer_name: str) -> str:
    return layer_name.replace(".", "_").replace("/", "_")


def init_h5_files(out_dir: Path, layer_names: list[str], n_frags: int, hidden_dim: int):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragments", default="processed_data/fragments.tsv")
    parser.add_argument("--out_dir",   default="processed_data/embeddings/evo2")
    parser.add_argument("--device",    default="cuda:0")
    parser.add_argument(
        "--layers",
        default="blocks.0.mlp.l3,blocks.16.mlp.l3,blocks.24.mlp.l3,"
                "blocks.26.mlp.l3,blocks.28.mlp.l3,blocks.31.mlp.l3",
        help="Comma-separated layer NAMES. Default spans the network: "
             "0 (input), 16 (mid), 24/26/28 (paper-highlighted region), "
             "31 (final). The 7B model has 32 blocks (0..31).",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_names = [s.strip() for s in args.layers.split(",") if s.strip()]
    if not layer_names:
        raise ValueError("--layers parsed to an empty list.")
    print(f"Saving embeddings for layers: {layer_names}")

    # --- Load fragments ---
    print("Loading fragments ...")
    df = pd.read_csv(args.fragments, sep="\t")
    sequences = df["seq"].tolist()
    frag_ids  = df["frag_id"].tolist()
    n_frags   = len(df)
    print(f"  {n_frags} fragments to embed")

    # --- Load model ---
    evo2_model = load_evo2()

    # --- Discover hidden_dim with a dummy forward pass ---
    print("Probing hidden dimension ...")
    probe = extract_layer_embeddings("ACGT", evo2_model, layer_names, args.device)
    hidden_dim = next(iter(probe.values())).shape[-1]
    print(f"  hidden_dim = {hidden_dim}")

    # --- Write frag_ids ---
    (out_dir / "frag_ids.txt").write_text("\n".join(frag_ids))

    # --- Resume offset ---
    start_row = 0
    if args.resume:
        cursor_path = out_dir / ".cursor"
        if cursor_path.exists():
            start_row = int(cursor_path.read_text().strip())
            print(f"Resuming from row {start_row}")

    # --- Init HDF5 files (only if starting fresh) ---
    if start_row == 0:
        h5_handles = init_h5_files(out_dir, layer_names, n_frags, hidden_dim)
    else:
        h5_handles = {
            name: h5py.File(out_dir / f"{safe_filename(name)}.h5", "r+")
            for name in layer_names
        }

    # --- Per-sequence forward passes ---
    print("\nExtracting embeddings ...")
    for i in range(start_row, n_frags):
        try:
            per_layer = extract_layer_embeddings(
                sequences[i], evo2_model, layer_names, args.device,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at fragment {i}. Try a shorter fragment length.")
            raise

        for name, vec in per_layer.items():
            h5_handles[name]["embeddings"][i] = vec

        (out_dir / ".cursor").write_text(str(i + 1))

        if i % 100 == 0 or i == n_frags - 1:
            pct = 100 * (i + 1) / n_frags
            print(f"  {i + 1}/{n_frags} ({pct:.1f}%)  ", end="\r")

        if i % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nDone. Embeddings saved to {out_dir}")
    close_h5_files(h5_handles)

    # Sanity check
    first = safe_filename(layer_names[0])
    with h5py.File(out_dir / f"{first}.h5", "r") as fh:
        print(f"{first}.h5 shape: {fh['embeddings'].shape}")


if __name__ == "__main__":
    main()