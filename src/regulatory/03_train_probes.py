"""
03_train_probes.py
------------------
Trains classification probes on cached frozen-LM embeddings for Task 1
(gene_body vs regulatory).

Pipeline position
-----------------
  00_download_data.sh -> 01_build_windows.py -> ../extract_embeddings.py
  -> 03_train_probes.py (this) -> 05_umap_visualization.py

Inputs
------
  windows.tsv from 01_build_windows.py
      columns: window_id, chrom, start, end, label, split, gc_content, seq
      label in {gene_body, regulatory}; split in {train, val, test, other}
  embeddings from ../extract_embeddings.py
      <embeddings_root>/<model>/layer_NN.h5  (dataset "embeddings", shape (N, H))
      <embeddings_root>/<model>/window_ids.txt  (row order of the H5 files)

Embeddings are aligned to windows.tsv by window_id (via window_ids.txt),
so row order is robust even if the two were written separately.

Probes (per model, per layer)
------------------------------
  - L2 logistic regression (C tuned on the val split)
  - shallow MLP (1 hidden layer, early stopping)
  Features standardized with StandardScaler (fit on train).

Metrics (binary, positive class = regulatory)
  F1, ROC-AUC, AUPRC (average precision).

Outputs
-------
  <out_dir>/classification_results.tsv
      (model, layer, layer_idx, probe, f1, auc, auprc, n_train, n_test)
  <out_dir>/layer_curves.png         (metric vs layer, one line per model x probe)
  <out_dir>/single_layer_summary.png (bar chart when only one layer is present)

Usage
-----
  # all models found under data/embeddings, all layers
  python 03_train_probes.py
  # one model, logistic only
  python 03_train_probes.py --models evo2 --skip_mlp
  # specific layers
  python 03_train_probes.py --probe_layers 0,8,16
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


POS_LABEL = "regulatory"  # positive class for binary AUC / AUPRC
_LAYER_RE = re.compile(r"layer_(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _layer_index(path: Path) -> int:
    m = _LAYER_RE.search(path.stem)
    return int(m.group(1)) if m else 10**9


def discover_models(root: Path, requested: str) -> dict[str, Path]:
    """Map model name -> its embeddings dir (must contain layer_*.h5)."""
    found: dict[str, Path] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(d.glob("layer_*.h5")):
            found[d.name] = d
    if requested.strip().lower() == "all":
        return found
    wanted = [m.strip() for m in requested.split(",") if m.strip()]
    missing = [m for m in wanted if m not in found]
    if missing:
        raise FileNotFoundError(
            f"Requested models not found under {root}: {missing}. "
            f"Available: {sorted(found)}"
        )
    return {m: found[m] for m in wanted}


def discover_layer_files(model_dir: Path) -> dict[str, Path]:
    """Map display layer name 'layer_NN' -> H5 path, ordered by index."""
    files = sorted(model_dir.glob("layer_*.h5"), key=_layer_index)
    return {p.stem: p for p in files}


def resolve_requested_layers(requested: str, available: dict[str, Path]) -> list[str]:
    if requested.strip().lower() == "all":
        return list(available.keys())
    # Accept either "layer_08" or bare "8".
    by_idx = {str(_layer_index(p)): name for name, p in available.items()}
    resolved, missing = [], []
    for tok in (s.strip() for s in requested.split(",") if s.strip()):
        if tok in available:
            resolved.append(tok)
        elif tok in by_idx:
            resolved.append(by_idx[tok])
        else:
            missing.append(tok)
    if missing:
        raise FileNotFoundError(
            f"Requested layers not found: {missing}. Available: {list(available)}"
        )
    seen, ordered = set(), []
    for name in resolved:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


# ---------------------------------------------------------------------------
# Alignment & loading
# ---------------------------------------------------------------------------

def build_label_split_arrays(windows: pd.DataFrame, model_dir: Path):
    """
    Align windows.tsv to the embedding row order recorded in window_ids.txt.

    Returns (y, split, row_keep) where:
      y         : str label per embedding row (NaN-free, aligned)
      split     : str split per embedding row
      row_keep  : bool mask over embedding rows that matched a window_id
    """
    ids_path = model_dir / "window_ids.txt"
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing {ids_path} (written by extract_embeddings.py)")
    emb_ids = ids_path.read_text().splitlines()

    meta = windows.set_index("window_id")
    row_keep = np.array([wid in meta.index for wid in emb_ids])
    if not row_keep.all():
        n_miss = int((~row_keep).sum())
        print(f"    WARNING: {n_miss}/{len(emb_ids)} embedding rows have no "
              f"matching window_id in windows.tsv — dropping them.")
    kept_ids = [wid for wid, keep in zip(emb_ids, row_keep) if keep]
    sub = meta.loc[kept_ids]
    return sub["label"].values, sub["split"].values, row_keep


def load_split(layer_path: Path, row_keep: np.ndarray, split_arr: np.ndarray,
               split: str) -> np.ndarray:
    """Read embeddings for one split, honoring the row_keep alignment mask."""
    with h5py.File(layer_path, "r") as fh:
        X = fh["embeddings"][:]            # (N_emb, H)
    X = X[row_keep]                        # now aligned with split_arr / y
    return X[split_arr == split]


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def train_probe(X_tr, y_tr, X_va, y_va, X_te, y_te, probe_type: str) -> dict:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    if probe_type == "logistic":
        best_f1, clf = -1.0, None
        for C in [0.01, 0.1, 1.0, 10.0]:
            m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", n_jobs=-1)
            m.fit(X_tr_s, y_tr)
            val_f1 = f1_score(y_va, m.predict(X_va_s),
                              pos_label=POS_LABEL, zero_division=0)
            if val_f1 > best_f1:
                best_f1, clf = val_f1, m
    elif probe_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256,), activation="relu", alpha=1e-3,
            max_iter=200, early_stopping=True, validation_fraction=0.1,
            random_state=42,
        )
        clf.fit(X_tr_s, y_tr)
    else:
        raise ValueError(probe_type)

    y_pred = clf.predict(X_te_s)
    f1 = f1_score(y_te, y_pred, pos_label=POS_LABEL, zero_division=0)

    auc = auprc = float("nan")
    if POS_LABEL in clf.classes_ and len(np.unique(y_te)) > 1:
        pos_col = list(clf.classes_).index(POS_LABEL)
        scores = clf.predict_proba(X_te_s)[:, pos_col]
        y_true_bin = (y_te == POS_LABEL).astype(int)
        auc = roc_auc_score(y_true_bin, scores)
        auprc = average_precision_score(y_true_bin, scores)

    return {"f1": float(f1), "auc": float(auc), "auprc": float(auprc)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_layer_curves(df: pd.DataFrame, out_path: Path):
    if not HAS_MPL or df.empty or df["layer_idx"].nunique() < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for (model, probe), sub in df.groupby(["model", "probe"]):
        sub = sub.sort_values("layer_idx")
        for ax, metric in zip(axes, ["f1", "auc", "auprc"]):
            ax.plot(sub["layer_idx"], sub[metric], marker="o",
                    label=f"{model}/{probe}")
    for ax, metric in zip(axes, ["f1", "auc", "auprc"]):
        ax.set_xlabel("Layer index")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Layer-wise {metric.upper()}")
        ax.legend(fontsize=7)
    fig.suptitle("Task 1 (gene_body vs regulatory) — probe metrics by layer")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_single_layer(df: pd.DataFrame, out_path: Path):
    if not HAS_MPL or df.empty:
        return
    metrics = ["f1", "auc", "auprc"]
    labels = [f"{r.model}/{r.layer}/{r.probe}" for r in df.itertuples()]
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(df))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] if np.isfinite(row[m]) else 0.0 for m in metrics]
        ax.bar(x + i * width, vals, width=width, label=labels[i])
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)
    ax.set_title("Task 1 — single-layer probe summary")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", default="data/regulatory/windows.tsv")
    p.add_argument("--embeddings_root", default="data/embeddings")
    p.add_argument("--models", default="all",
                   help="Comma-separated model subdir names, or 'all'.")
    p.add_argument("--probe_layers", default="all",
                   help="Comma-separated layer indices/names, or 'all'.")
    p.add_argument("--out_dir", default="data/regulatory/results")
    p.add_argument("--skip_mlp", action="store_true",
                   help="Only run logistic regression (faster).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows, sep="\t")
    for col in ("window_id", "label", "split"):
        if col not in windows.columns:
            raise ValueError(f"windows.tsv missing required column '{col}'")
    print(f"{len(windows)} windows; label counts:\n"
          f"{windows['label'].value_counts().to_string()}")

    models = discover_models(Path(args.embeddings_root), args.models)
    if not models:
        raise FileNotFoundError(
            f"No model dirs with layer_*.h5 under {args.embeddings_root}. "
            f"Run ../extract_embeddings.py first."
        )
    print(f"Models: {list(models)}")

    probe_types = ["logistic"] if args.skip_mlp else ["logistic", "mlp"]
    rows = []

    for model, model_dir in models.items():
        print(f"\n########## model: {model} ({model_dir}) ##########")
        y, split_arr, row_keep = build_label_split_arrays(windows, model_dir)

        available = discover_layer_files(model_dir)
        layers = resolve_requested_layers(args.probe_layers, available)
        print(f"  layers: {layers}")

        for layer in layers:
            layer_path = available[layer]
            X_tr = load_split(layer_path, row_keep, split_arr, "train")
            X_va = load_split(layer_path, row_keep, split_arr, "val")
            X_te = load_split(layer_path, row_keep, split_arr, "test")
            y_tr = y[split_arr == "train"]
            y_va = y[split_arr == "val"]
            y_te = y[split_arr == "test"]

            if min(len(X_tr), len(X_va), len(X_te)) == 0:
                print(f"  {layer}: empty split (train/val/test = "
                      f"{len(X_tr)}/{len(X_va)}/{len(X_te)}) — skipping.")
                continue
            if not np.isfinite(X_tr).all():
                raise ValueError(
                    f"{model}/{layer}: non-finite embeddings in train split — "
                    f"re-run extract_embeddings.py."
                )

            for probe in probe_types:
                m = train_probe(X_tr, y_tr, X_va, y_va, X_te, y_te, probe)
                print(f"  {layer:>10s} {probe:8s} "
                      f"F1={m['f1']:.3f} AUC={m['auc']:.3f} AUPRC={m['auprc']:.3f}")
                rows.append({
                    "model": model, "layer": layer,
                    "layer_idx": _layer_index(layer_path),
                    "probe": probe, **m,
                    "n_train": len(X_tr), "n_test": len(X_te),
                })

    res = pd.DataFrame(rows)
    res_path = out_dir / "classification_results.tsv"
    res.to_csv(res_path, sep="\t", index=False)
    print(f"\nResults saved to {res_path}")

    if res.empty:
        return
    if res["layer_idx"].nunique() >= 2:
        plot_layer_curves(res, out_dir / "layer_curves.png")
    else:
        plot_single_layer(res, out_dir / "single_layer_summary.png")

    print("\n=== Best (logistic, by AUPRC) per model ===")
    log = res[res["probe"] == "logistic"]
    for model, sub in log.groupby("model"):
        best = sub.sort_values("auprc").iloc[-1]
        print(f"  {model}: {best['layer']} -> "
              f"AUPRC={best['auprc']:.3f}, AUC={best['auc']:.3f}, F1={best['f1']:.3f}")


if __name__ == "__main__":
    main()
