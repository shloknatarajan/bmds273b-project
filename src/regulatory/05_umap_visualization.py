"""
05_umap_visualization.py
------------------------
UMAP projections of frozen-LM embeddings for Task 1, colored by class
(gene_body vs regulatory) and by GC content. The GC plot is a sanity
check: after GC-matched sampling in 01_build_windows.py, class structure
should NOT simply track GC.

Consumes the same cached embeddings as 03_train_probes.py
(<embeddings_dir>/layer_NN.h5 + window_ids.txt), aligned to windows.tsv
by window_id.

Outputs
-------
  <out_dir>/<model>_layer_NN_label.png
  <out_dir>/<model>_layer_NN_gc.png

Usage
-----
  python 05_umap_visualization.py --embeddings_dir data/embeddings/evo2
  python 05_umap_visualization.py --embeddings_dir data/embeddings/nt --layers 8,16
"""

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import umap  # umap-learn

_LAYER_RE = re.compile(r"layer_(\d+)", re.IGNORECASE)
LABEL_PALETTE = {"gene_body": "#4daf4a", "regulatory": "#e41a1c"}


def _layer_index(path: Path) -> int:
    m = _LAYER_RE.search(path.stem)
    return int(m.group(1)) if m else 10**9


def discover_layer_files(emb_dir: Path) -> dict[str, Path]:
    files = sorted(emb_dir.glob("layer_*.h5"), key=_layer_index)
    return {p.stem: p for p in files}


def resolve_layers(layers_arg: str, available: dict[str, Path]) -> list[str]:
    if layers_arg.strip().lower() == "all":
        return list(available.keys())
    by_idx = {str(_layer_index(p)): name for name, p in available.items()}
    resolved, missing = [], []
    for tok in (s.strip() for s in layers_arg.split(",") if s.strip()):
        if tok in available:
            resolved.append(tok)
        elif tok in by_idx:
            resolved.append(by_idx[tok])
        else:
            missing.append(tok)
    if missing:
        print(f"  WARNING: layers not found, skipping: {missing}")
    seen, ordered = set(), []
    for name in resolved:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def align_rows(windows: pd.DataFrame, emb_dir: Path):
    """Return (meta_df, row_keep) aligning embedding rows to windows.tsv."""
    emb_ids = (emb_dir / "window_ids.txt").read_text().splitlines()
    meta = windows.set_index("window_id")
    row_keep = np.array([wid in meta.index for wid in emb_ids])
    kept_ids = [wid for wid, k in zip(emb_ids, row_keep) if k]
    return meta.loc[kept_ids].reset_index(), row_keep


def run_umap(X, n_neighbors, min_dist, seed):
    X_s = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed)
    return reducer.fit_transform(X_s)


def plot_categorical(emb2d, labels, palette, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    for lbl in pd.unique(labels):
        m = labels == lbl
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=8, alpha=0.6,
                   c=palette.get(lbl, "#888888"), label=str(lbl), rasterized=True)
    ax.legend(markerscale=2, fontsize=9)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_continuous(emb2d, values, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], s=8, alpha=0.6,
                    c=values, cmap="viridis", rasterized=True)
    fig.colorbar(sc, ax=ax, label="GC content")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", default="data/regulatory/windows.tsv")
    p.add_argument("--embeddings_dir", default="data/embeddings/evo2")
    p.add_argument("--out_dir", default="data/regulatory/results/umap")
    p.add_argument("--layers", default="all")
    p.add_argument("--split", default="test")
    p.add_argument("--max_points", type=int, default=5000)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.embeddings_dir)
    model = emb_dir.name

    available = discover_layer_files(emb_dir)
    if not available:
        raise FileNotFoundError(f"No layer_*.h5 in {emb_dir}. Run extract_embeddings.py.")
    layers = resolve_layers(args.layers, available)
    if not layers:
        raise RuntimeError(f"None of the requested layers present. Available: {list(available)}")
    print(f"Model={model}; visualizing layers: {layers}")

    windows = pd.read_csv(args.windows, sep="\t")
    meta, row_keep = align_rows(windows, emb_dir)

    # Choose rows in the requested split, subsampled for UMAP speed.
    split_pos = np.where(meta["split"].values == args.split)[0]
    rng = np.random.default_rng(args.seed)
    if len(split_pos) > args.max_points:
        split_pos = rng.choice(split_pos, args.max_points, replace=False)
    split_pos.sort()
    labels = meta["label"].values[split_pos]
    gc = meta["gc_content"].values[split_pos] if "gc_content" in meta else None
    print(f"  {len(split_pos)} {args.split} windows "
          f"({np.sum(labels=='regulatory')} regulatory / "
          f"{np.sum(labels=='gene_body')} gene_body)")

    for layer in layers:
        print(f"\n=== UMAP {model}/{layer} ===")
        with h5py.File(available[layer], "r") as fh:
            X = fh["embeddings"][:][row_keep][split_pos].astype(np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f"{model}/{layer}: non-finite embeddings.")
        emb2d = run_umap(X, args.n_neighbors, args.min_dist, args.seed)

        stem = f"{model}_{layer}"
        plot_categorical(emb2d, labels, LABEL_PALETTE,
                         f"{model} — {layer} — class ({args.split})",
                         out_dir / f"{stem}_label.png")
        if gc is not None:
            plot_continuous(emb2d, gc,
                            f"{model} — {layer} — GC content ({args.split})",
                            out_dir / f"{stem}_gc.png")

    print("\nUMAP visualizations complete.")


if __name__ == "__main__":
    main()
