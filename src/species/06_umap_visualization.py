"""
06_umap_visualization.py
------------------------
Generates UMAP projections of Evo 2 embeddings at selected layers,
colored by phylum and by domain (Bacteria / Archaea).

Run after 05_train_probes.py — it uses the same cached HDF5 embeddings.

Outputs
-------
results/umap/
    layer_{i:02d}_phylum.png
    layer_{i:02d}_domain.png

Usage
-----
  python 06_umap_visualization.py \
      --embeddings_dir data/embeddings/evo2 \
      --fragments      data/fragments/fragments.tsv \
      --out_dir        results/umap \
      --layers         0,12,24   # embedding layer, mid, final
      --n_neighbors    15 \
      --min_dist       0.1
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:
    raise ImportError("Install umap-learn: pip install umap-learn")


PHYLUM_PALETTE = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#999999","#66c2a5","#fc8d62",
    "#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494",
    "#b3b3b3","#1b9e77","#d95f02","#7570b3","#e7298a",
]

DOMAIN_PALETTE = {"Bacteria": "#4393c3", "Archaea": "#d6604d"}


def load_layer_embeddings(emb_dir: Path, layer_idx: int, row_mask: np.ndarray) -> np.ndarray:
    with h5py.File(emb_dir / f"layer_{layer_idx:02d}.h5", "r") as fh:
        rows = np.where(row_mask)[0]
        X = fh["embeddings"][rows, :]
    return X.astype(np.float32)


def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float, seed: int = 42):
    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed, verbose=False)
    return reducer.fit_transform(X_scaled)


def plot_umap(embedding_2d: np.ndarray, labels: np.ndarray,
              palette: dict | list, title: str, out_path: Path):
    unique_labels = pd.unique(labels)
    if isinstance(palette, list):
        color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}
    else:
        color_map = palette

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=color_map.get(lbl, "#888888"),
            s=8, alpha=0.7, label=lbl, rasterized=True,
        )

    # Compact legend outside plot
    n_cols = max(1, len(unique_labels) // 20)
    ax.legend(
        markerscale=2, fontsize=7, ncol=n_cols,
        bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", default="data/embeddings/evo2")
    parser.add_argument("--fragments",      default="data/fragments/fragments.tsv")
    parser.add_argument("--out_dir",        default="results/umap")
    parser.add_argument("--layers",         default="0",
                        help="Comma-separated layer indices to visualize. "
                             "E.g. '0,12,24' for embedding, mid, and final layers.")
    parser.add_argument("--split",          default="test",
                        help="Which split to visualize (test recommended).")
    parser.add_argument("--max_points",     type=int, default=5000,
                        help="Subsample to this many points for UMAP speed.")
    parser.add_argument("--n_neighbors",    type=int, default=15)
    parser.add_argument("--min_dist",       type=float, default=0.1)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.embeddings_dir)

    df = pd.read_csv(args.fragments, sep="\t")
    split_mask = (df["split"] == args.split).values

    # Optional subsampling for speed
    split_indices = np.where(split_mask)[0]
    rng = np.random.default_rng(args.seed)
    if len(split_indices) > args.max_points:
        chosen = rng.choice(split_indices, size=args.max_points, replace=False)
        chosen_mask = np.zeros(len(df), dtype=bool)
        chosen_mask[chosen] = True
    else:
        chosen_mask = split_mask

    phylum_labels = df.loc[chosen_mask, "phylum"].values
    domain_labels = df.loc[chosen_mask, "domain"].values

    layer_indices = [int(x) for x in args.layers.split(",")]

    for layer_idx in layer_indices:
        print(f"\n=== UMAP for layer {layer_idx} ===")
        X = load_layer_embeddings(emb_dir, layer_idx, chosen_mask)
        print(f"  Embedding shape: {X.shape}")
        print("  Running UMAP ...")
        emb_2d = run_umap(X, args.n_neighbors, args.min_dist, args.seed)

        plot_umap(
            emb_2d, phylum_labels, PHYLUM_PALETTE,
            title=f"Evo 2 7B — Layer {layer_idx} — Colored by Phylum ({args.split} set)",
            out_path=out_dir / f"layer_{layer_idx:02d}_phylum.png",
        )
        plot_umap(
            emb_2d, domain_labels, DOMAIN_PALETTE,
            title=f"Evo 2 7B — Layer {layer_idx} — Colored by Domain ({args.split} set)",
            out_path=out_dir / f"layer_{layer_idx:02d}_domain.png",
        )

    print("\nUMAP visualizations complete.")


if __name__ == "__main__":
    main()
