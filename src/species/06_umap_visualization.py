"""
06_umap_visualization.py
------------------------
Generates UMAP projections of Evo 2 embeddings at selected layers,
colored by phylum and by domain (Bacteria / Archaea).

Run after 04_extract_embeddings_evo2.py — it uses the same cached
HDF5 embeddings.

By default we plot a curated set of layers that always includes
**layer 26** (the layer the Evo 2 paper highlights) plus a sampling of
other interesting depths (embedding layer, an early block, a mid block,
the final block). Pass ``--layers all`` to visualize every extracted
layer, or pass a comma-separated list to override the selection.

Outputs
-------
processed_data/results/umap/
    layer_{i:02d}_phylum.png
    layer_{i:02d}_domain.png

Usage
-----
  # Default: 0, 16, 26, 32 (whichever of these are available)
  python 06_umap_visualization.py

  # All available layers (slow — one UMAP fit per layer)
  python 06_umap_visualization.py --layers all

  # Just the paper-highlighted layer
  python 06_umap_visualization.py --layers 26
"""

import argparse
import re
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


LAYER_FILE_RE = re.compile(r"layer_(\d+)\.h5$")

# Default curated layers — embedding, early, mid, paper-highlighted, final
DEFAULT_LAYERS = "0,16,26,32"

# Layer highlighted by the Evo 2 paper as the most informative single layer.
PAPER_LAYER = 26


PHYLUM_PALETTE = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#999999","#66c2a5","#fc8d62",
    "#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494",
    "#b3b3b3","#1b9e77","#d95f02","#7570b3","#e7298a",
]

DOMAIN_PALETTE = {"Bacteria": "#4393c3", "Archaea": "#d6604d"}


def discover_layer_files(emb_dir: Path) -> dict[int, Path]:
    """Map layer index -> HDF5 path for every layer file present."""
    out: dict[int, Path] = {}
    for p in emb_dir.glob("layer_*.h5"):
        m = LAYER_FILE_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def resolve_layers(layers_arg: str, available: dict[int, Path]) -> list[int]:
    """
    Turn the --layers CLI argument into an explicit ordered list of
    indices that exist in ``available``. Requested layers that have not
    been extracted are skipped with a warning.
    """
    if layers_arg.strip().lower() == "all":
        return sorted(available.keys())

    requested = [int(x) for x in layers_arg.split(",") if x.strip()]
    resolved: list[int] = []
    for idx in requested:
        if idx in available:
            resolved.append(idx)
        else:
            print(f"  WARNING: layer {idx} not present in embeddings dir; skipping.")
    return resolved


def load_layer_embeddings(layer_path: Path, row_mask: np.ndarray) -> np.ndarray:
    with h5py.File(layer_path, "r") as fh:
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
    parser.add_argument("--embeddings_dir", default="processed_data/embeddings/evo2")
    parser.add_argument("--fragments",      default="processed_data/fragments.tsv")
    parser.add_argument("--out_dir",        default="processed_data/results/umap")
    parser.add_argument("--layers",         default=DEFAULT_LAYERS,
                        help=f"Comma-separated layer indices, or 'all'. "
                             f"Default: '{DEFAULT_LAYERS}'. Layer "
                             f"{PAPER_LAYER} is the paper-highlighted one.")
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

    available = discover_layer_files(emb_dir)
    if not available:
        raise FileNotFoundError(
            f"No layer_*.h5 files found in {emb_dir}. "
            f"Run 04_extract_embeddings_evo2.py first."
        )
    print(f"Found {len(available)} layer file(s) in {emb_dir}: "
          f"{sorted(available.keys())}")

    layer_indices = resolve_layers(args.layers, available)
    if not layer_indices:
        raise RuntimeError(
            f"None of the requested layers ({args.layers}) are present. "
            f"Available: {sorted(available.keys())}."
        )
    print(f"Visualizing layers: {layer_indices}")

    df = pd.read_csv(args.fragments, sep="\t")
    split_mask = (df["split"] == args.split).values

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

    for layer_idx in layer_indices:
        layer_path = available[layer_idx]
        tag = " [paper highlight]" if layer_idx == PAPER_LAYER else ""
        print(f"\n=== UMAP for layer {layer_idx}{tag} ===")
        X = load_layer_embeddings(layer_path, chosen_mask)
        print(f"  Embedding shape: {X.shape}")
        print("  Running UMAP ...")
        emb_2d = run_umap(X, args.n_neighbors, args.min_dist, args.seed)

        title_suffix = " — Evo 2 paper layer" if layer_idx == PAPER_LAYER else ""
        plot_umap(
            emb_2d, phylum_labels, PHYLUM_PALETTE,
            title=(f"Evo 2 7B — Layer {layer_idx}{title_suffix} — "
                   f"Phylum ({args.split} set)"),
            out_path=out_dir / f"layer_{layer_idx:02d}_phylum.png",
        )
        plot_umap(
            emb_2d, domain_labels, DOMAIN_PALETTE,
            title=(f"Evo 2 7B — Layer {layer_idx}{title_suffix} — "
                   f"Domain ({args.split} set)"),
            out_path=out_dir / f"layer_{layer_idx:02d}_domain.png",
        )

    print("\nUMAP visualizations complete.")


if __name__ == "__main__":
    main()
