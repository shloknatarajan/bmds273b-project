"""
Extract per-layer embeddings from Evo 2 7B via Arc Institute's native `evo2`
package (the HuggingFace AutoModel path is broken for Evo 2).

Evo 2 7B = StripedHyena 2, 32 blocks. We sample a spread of blocks and mean-pool
the `blocks.N.mlp.l3` activation per window. Output mirrors the HF extractor:
one layer_{NN}.h5 per sampled block (NN = block index) so train_probes.py /
umap_visualization.py treat Evo 2 like any other model.

Usage:
    python src/extract_embeddings_evo2.py --windows data/regulatory/windows.tsv \
        --out_dir data/regulatory/embeddings/evo2
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from evo2 import Evo2


# Spread of blocks across the 32-block network (→ layer_03.h5 .. layer_31.h5).
DEFAULT_BLOCKS = [3, 7, 11, 15, 19, 23, 27, 31]
MAX_TOK = 8_192  # Evo 2 7B context used here


def block_layer_name(b: int) -> str:
    return f"blocks.{b}.mlp.l3"


@torch.inference_mode()
def embed_one(seq, model, block_names, device):
    ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.long).unsqueeze(0).to(device)
    if ids.shape[1] > MAX_TOK:
        ids = ids[:, :MAX_TOK]
    _, emb = model(ids, return_embeddings=True, layer_names=block_names)
    out = []
    for name in block_names:
        hs = emb[name].float()              # (1, seq_len, hidden)
        out.append(hs.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", default="data/regulatory/windows.tsv")
    p.add_argument("--id_col", default="window_id")
    p.add_argument("--out_dir", default="data/regulatory/embeddings/evo2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--blocks", default=",".join(map(str, DEFAULT_BLOCKS)))
    p.add_argument("--model", default="evo2_7b")
    args = p.parse_args()

    blocks = [int(b) for b in args.blocks.split(",") if b.strip() != ""]
    block_names = [block_layer_name(b) for b in blocks]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.windows, sep="\t")
    seqs = df["seq"].tolist()
    window_ids = df[args.id_col].astype(str).tolist()
    n = len(df)
    print(f"{n} windows; Evo 2 blocks {blocks}")

    print(f"Loading {args.model} via evo2 package ...")
    model = Evo2(args.model)
    model.model.eval()
    for prm in model.model.parameters():
        prm.requires_grad = False

    (out_dir / "window_ids.txt").write_text("\n".join(window_ids))

    # probe for hidden_dim
    probe = embed_one(seqs[0], model, block_names, args.device)
    hidden_dim = probe[0].shape[0]
    print(f"  hidden_dim={hidden_dim}")

    handles = {}
    for b in blocks:
        path = out_dir / f"layer_{b:02d}.h5"
        if path.exists():
            path.unlink()
        fh = h5py.File(path, "w")
        fh.create_dataset("embeddings", shape=(n, hidden_dim), dtype="float32",
                          chunks=(min(256, n), hidden_dim))
        handles[b] = fh

    BLOCK = 256
    bufs = np.zeros((len(blocks), BLOCK, hidden_dim), dtype=np.float32)

    def flush(start, count):
        if count == 0:
            return
        for k, b in enumerate(blocks):
            handles[b]["embeddings"][start:start + count] = bufs[k, :count]

    buf_start, fill = 0, 0
    for row in range(n):
        vecs = embed_one(seqs[row], model, block_names, args.device)
        for k in range(len(blocks)):
            bufs[k, fill] = vecs[k]
        fill += 1
        if fill == BLOCK:
            flush(buf_start, fill)
            buf_start += fill
            fill = 0
            torch.cuda.empty_cache()
        if row % 100 == 0 or row == n - 1:
            print(f"  {row + 1}/{n} ({100*(row+1)/n:.1f}%)", end="\r")

    flush(buf_start, fill)
    for fh in handles.values():
        fh.close()
    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
