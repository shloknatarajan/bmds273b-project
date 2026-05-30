"""
06_umap_visualization.py
------------------------
Generates UMAP projections of Nucleotide Transformer embeddings at selected layers,
colored by phylum and by domain (Bacteria / Archaea).

Run after 04_extract_embeddings_nt.py - it consumes the same cached
HDF5 embeddings. Script 04 saves one HDF5 per requested layer using a
safe-filename convention (e.g. ``blocks.28.mlp.l3`` -> ``blocks_28_mlp_l3.h5``).
The original layer name is preserved in the file's ``layer_name``
attribute and used as the display label here.

Outputs
-------
processed_data/results/umap/
    <safe_layer_name>_phylum.png
    <safe_layer_name>_domain.png

Usage
-----
  # All available layers (default)
  python 06_umap_visualization.py

  # Specific layers (dotted or underscored form, comma-separated)
  python 06_umap_visualization.py --layers blocks.28.mlp.l3
  python 06_umap_visualization.py --layers blocks_15_mlp_l3,blocks_28_mlp_l3
"""

import argparse
from pathlib import Path
import re
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import umap  # umap-learn; needs numba which requires NumPy <= 2.1


PHYLUM_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
]

DOMAIN_PALETTE = {"Bacteria": "#4393c3", "Archaea": "#d6604d"}

# ---------------------------------------------------------------------------
# Layer-file discovery
# ---------------------------------------------------------------------------


def _block_index(layer_name: str) -> int:
    m = re.search(r"layer[._](\d+)", layer_name)
    if m:
        return int(m.group(1))
    return 10 ** 9

def _layer_sort_key(layer_name: str):
    return (_block_index(layer_name), layer_name)
    
def canonicalize_layer_name(name: str) -> str:
    """
    Convert safe-filename variants back to dotted canonical form.

    Examples:
      layer_12  -> layer.12
      layer.12  -> layer.12   (already canonical, unchanged)
    """
    # safe-filename form: layer_N
    m = re.fullmatch(r"layer_(\d+)", name)
    if m:
        return f"layer.{int(m.group(1))}"
    # already dotted
    m = re.fullmatch(r"layer\.(\d+)", name)
    if m:
        return f"layer.{int(m.group(1))}"
    return name

def discover_layer_files(emb_dir: Path) -> dict[str, Path]:
    """
    Returns:
      {
        canonical_layer_name: path_to_h5
      }

    Reads the ``layer_name`` attribute stored by 04_extract_embeddings_nt_v2.py
    (e.g. 'layer.12') and maps it back to the canonical form.
    """
    found = {}

    for p in sorted(emb_dir.glob("*.h5")):
        try:
            with h5py.File(p, "r") as fh:
                raw_name = fh.attrs.get("layer_name", p.stem)
        except OSError:
            continue

        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode()

        canonical = canonicalize_layer_name(str(raw_name))

        # keep first occurrence only (avoids duplicates if files are re-run)
        if canonical not in found:
            found[canonical] = p

    return dict(sorted(found.items(), key=lambda kv: _layer_sort_key(kv[0])))

def safe_filename(layer_name: str) -> str:
    return layer_name.replace(".", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_layer_embeddings(layer_path: Path, row_mask: np.ndarray) -> np.ndarray:
    with h5py.File(layer_path, "r") as fh:
        rows = np.where(row_mask)[0]
        X = fh["embeddings"][rows, :]
    return X.astype(np.float32)


def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float, seed: int = 42):
    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, random_state=seed, verbose=False,
    )
    return reducer.fit_transform(X_scaled)


def plot_umap(
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    palette: dict | list,
    title: str,
    out_path: Path,
):
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
    parser.add_argument("--embeddings_dir", default="processed_data/embeddings/nt")
    parser.add_argument("--fragments",      default="processed_data/fragments.tsv")
    parser.add_argument("--out_dir",        default="processed_data/results/umap")
    parser.add_argument(
        "--layers",
        default="layer.1,layer.12,layer.24,layer.32",
        help=(
            "Comma-separated layer names in dotted form (layer.N) or "
            "underscored form (layer_N), or 'all' to use every cached layer. "
        ),
    )
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
            f"No .h5 files found in {emb_dir}. "
            f"Run 04_extract_embeddings_nt.py first."
        )
    print(f"----------\nFound {len(available)} layer file(s) in {emb_dir}: "
          f"{list(available.keys())}")

    if args.layers.strip().lower() == "all":
        layer_names = list(available.keys())
    else:
        layer_names = [
            canonicalize_layer_name(s.strip())
            for s in args.layers.split(",")
            if s.strip()
        ]

    missing = [x for x in layer_names if x not in available]
    if missing:
        raise ValueError(
            f"Missing layers: {missing}\n"
            f"Available layers: {list(available.keys())}"
        )

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
    print(f"  {chosen_mask.sum()} fragments, "
          f"{len(pd.unique(phylum_labels))} phyla, "
          f"{len(pd.unique(domain_labels))} domains")

    for layer_name in layer_names:
        layer_path = available[layer_name]
        print(f"\n=== UMAP for {layer_name} ({layer_path.name}) ===")
        X = load_layer_embeddings(layer_path, chosen_mask)
        print(f"  Embedding shape: {X.shape}")

        if not np.isfinite(X).all():
            n_bad = (~np.isfinite(X)).sum()
            raise ValueError(
                f"Embeddings for {layer_name} contain {n_bad} non-finite "
                f"values - re-run 04_extract_embeddings_nt.py."
            )

        print("  Running UMAP ...")
        emb_2d = run_umap(X, args.n_neighbors, args.min_dist, args.seed)

        stem = safe_filename(layer_name)
        plot_umap(
            emb_2d, phylum_labels, PHYLUM_PALETTE,
            title=f"Nucleotide Transformer - {layer_name} - Phylum ({args.split} set)",
            out_path=out_dir / f"{stem}_phylum.png",
        )
        plot_umap(
            emb_2d, domain_labels, DOMAIN_PALETTE,
            title=f"Nucleotide Transformer - {layer_name} - Domain ({args.split} set)",
            out_path=out_dir / f"{stem}_domain.png",
        )

    print("\nUMAP visualizations complete.")


if __name__ == "__main__":
    main()
