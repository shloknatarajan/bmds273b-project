"""
0umap_visualization.py
------------------------
UMAP projections of regulatory task embeddings, colored by
regulatory label (gene_body / regulatory) and by chromosome.

Run after 02_extract_embeddings.py.

Outputs
-------
  data/regulatory/results/umap/
    <model>_layer<idx>_label.png
    <model>_layer<idx>_chrom.png


"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


LABEL_PALETTE   = {"gene_body": "#4393c3", "regulatory": "#d6604d"}
SPLIT_PALETTE   = {"train": "#4daf4a", "val": "#ff7f00", "test": "#984ea3"}
CHROM_PALETTE   = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
]


# ---------------------------------------------------------------------------
# Discovery helpers (mirror 03_train_probes.py)
# ---------------------------------------------------------------------------

def discover_layers(model_dir: Path) -> dict[int, Path]:
    found = {}
    for p in sorted(model_dir.glob("layer_*.h5")):
        try:
            idx = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        found[idx] = p
    return dict(sorted(found.items()))


def discover_models(emb_root: Path) -> dict[str, dict[int, Path]]:
    models = {}
    for d in sorted(emb_root.iterdir()):
        if not d.is_dir():
            continue
        layers = discover_layers(d)
        if layers:
            models[d.name] = layers
    return models


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embeddings(layer_path: Path, row_mask: np.ndarray) -> np.ndarray:
    rows = np.where(row_mask)[0]
    with h5py.File(layer_path, "r") as fh:
        X = fh["embeddings"][rows, :].astype(np.float32)
    return X


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(
    X: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray:
    X_sc = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=seed,
        verbose=False,
    )
    return reducer.fit_transform(X_sc)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_umap(
    emb_2d: np.ndarray,
    labels: np.ndarray,
    palette,
    title: str,
    out_path: Path,
) -> None:
    unique = pd.unique(labels)
    if isinstance(palette, list):
        color_map = {lbl: palette[i % len(palette)]
                     for i, lbl in enumerate(sorted(unique))}
    else:
        color_map = dict(palette)

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl in sorted(unique):
        mask = labels == lbl
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=color_map.get(lbl, "#888888"),
            s=8, alpha=0.7, label=lbl, rasterized=True,
        )
    n_cols = max(1, len(unique) // 20)
    ax.legend(
        markerscale=2, fontsize=8, ncol=n_cols,
        bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_root", default="data/regulatory/embeddings")
    parser.add_argument("--windows",         default="data/regulatory/windows.tsv")
    parser.add_argument("--out_dir",         default="data/regulatory/results/umap")
    parser.add_argument(
        "--model", default="all",
        help="Model name, or 'all' to iterate every available model.",
    )
    parser.add_argument(
        "--layers", default="last",
        help=(
            "Comma-separated layer indices, 'all', or 'last' (default). "
            "'last' visualizes only the final layer of each model."
        ),
    )
    parser.add_argument("--split",       default="test",
                        help="Which data split to visualize.")
    parser.add_argument("--max_points",  type=int, default=5000,
                        help="Subsample to at most this many points for UMAP speed.")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist",    type=float, default=0.1)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if not HAS_UMAP:
        raise ImportError(
            "umap-learn is required. Install with: pip install umap-learn"
        )
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required. Install with: pip install matplotlib"
        )

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_root = Path(args.embeddings_root)

    # Discover models
    if args.model == "all":
        all_model_layers = discover_models(emb_root)
    else:
        d = emb_root / args.model
        layers = discover_layers(d)
        all_model_layers = {args.model: layers} if layers else {}

    if not all_model_layers:
        raise FileNotFoundError(
            f"No embeddings found in {emb_root}. Run 02_extract_embeddings.py first."
        )
    print(f"Models: {list(all_model_layers.keys())}")

    # Load windows and build subsample mask
    df         = pd.read_csv(args.windows, sep="\t")
    split_mask = (df["split"] == args.split).values

    rng = np.random.default_rng(args.seed)
    split_indices = np.where(split_mask)[0]
    if len(split_indices) > args.max_points:
        chosen_idx  = rng.choice(split_indices, size=args.max_points, replace=False)
        chosen_mask = np.zeros(len(df), dtype=bool)
        chosen_mask[chosen_idx] = True
    else:
        chosen_mask = split_mask

    label_labels = df.loc[chosen_mask, "label"].values
    chrom_labels  = df.loc[chosen_mask, "chrom"].values
    print(
        f"{chosen_mask.sum()} windows from '{args.split}' split  "
        f"({len(pd.unique(chrom_labels))} chroms)"
    )

    for model_name, layer_files in all_model_layers.items():
        print(f"\n=== {model_name} ===")

        if args.layers.strip().lower() == "all":
            layer_indices = sorted(layer_files.keys())
        elif args.layers.strip().lower() == "last":
            layer_indices = [max(layer_files.keys())]
        else:
            layer_indices = [
                int(l.strip()) for l in args.layers.split(",") if l.strip()
            ]
            layer_indices = [l for l in layer_indices if l in layer_files]

        if not layer_indices:
            print(f"  No matching layers found, skipping.")
            continue

        for layer_idx in layer_indices:
            layer_path = layer_files[layer_idx]
            print(f"  Layer {layer_idx:02d} ({layer_path.name}) ...")

            X = load_embeddings(layer_path, chosen_mask)
            if not np.isfinite(X).all():
                n_bad = (~np.isfinite(X)).sum()
                print(f"  WARNING: {n_bad} non-finite values — skipping.")
                continue

            print(f"  Running UMAP on shape {X.shape} ...")
            emb_2d = run_umap(X, args.n_neighbors, args.min_dist, args.seed)

            stem = f"{model_name}_layer{layer_idx:02d}"

            plot_umap(
                emb_2d, label_labels, LABEL_PALETTE,
                title=f"{model_name} layer {layer_idx} — Label ({args.split} set)",
                out_path=out_dir / f"{stem}_label.png",
            )
            plot_umap(
                emb_2d, chrom_labels, CHROM_PALETTE,
                title=f"{model_name} layer {layer_idx} — Chromosome ({args.split} set)",
                out_path=out_dir / f"{stem}_chrom.png",
            )

    print("\nUMAP visualizations complete.")


if __name__ == "__main__":
    main()
