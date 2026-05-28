"""
train_probes.py
------------------
Trains classification probes on frozen embeddings for the
regulatory vs. gene-body task (Task 1).

Setup
-----
  Per (model, layer):
    - L2 logistic regression  (C swept on val split)
    - Shallow MLP             (1 hidden layer, dropout, early stopping)
  Features are StandardScaler-normalised before each probe.

  Split (chromosome-level, from windows.tsv):
    train chr1-18 | val chr19-20 | test chr21-22 / X / Y

Metrics (binary: gene_body=0, regulatory=1)
-------------------------------------------
  F1, AUC, AUPRC on the test split.

Outputs
-------
  data/regulatory/results/
    probe_results.tsv           — (model, layer, probe, f1, auc, auprc)
    layer_curves_<model>.png    — metric vs layer index per model
    model_comparison.png        — best-layer AUPRC per model side-by-side

Usage
-----
  # All models, all layers
  python src/regulatory/03_train_probes.py

  # One model, subset of layers
  python src/regulatory/03_train_probes.py --model hyenadna --layers 0,8,16

  # Skip MLP for speed
  python src/regulatory/03_train_probes.py --skip_mlp
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Layer / model discovery
# ---------------------------------------------------------------------------

def discover_layers(model_dir: Path) -> dict[int, Path]:
    """Returns {layer_index: Path} for every layer_XX.h5 in model_dir."""
    found = {}
    for p in sorted(model_dir.glob("layer_*.h5")):
        try:
            idx = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        found[idx] = p
    return dict(sorted(found.items()))


def discover_models(emb_root: Path) -> dict[str, dict[int, Path]]:
    """Returns {model_name: {layer_idx: Path}} for all subdirs of emb_root."""
    models = {}
    for d in sorted(emb_root.iterdir()):
        if not d.is_dir():
            continue
        layers = discover_layers(d)
        if layers:
            models[d.name] = layers
    return models


def resolve_layer_indices(
    arg: str, available: dict[int, Path]
) -> list[int]:
    if arg.strip().lower() == "all":
        return sorted(available.keys())
    return [int(l.strip()) for l in arg.split(",") if l.strip()]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(
    df: pd.DataFrame,
    layer_path: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X float32, y int) for the requested split."""
    mask        = (df["split"] == split).values
    row_indices = np.where(mask)[0]
    with h5py.File(layer_path, "r") as fh:
        X = fh["embeddings"][row_indices, :].astype(np.float32)
    y = (df.loc[mask, "label"].values == "regulatory").astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_v:  np.ndarray, y_v:  np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    probe_type: str,
) -> dict:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_v_s  = scaler.transform(X_v)
    X_te_s = scaler.transform(X_te)

    if probe_type == "logistic":
        best_f1, best_clf = -1.0, None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            clf.fit(X_tr_s, y_tr)
            vf1 = f1_score(y_v, clf.predict(X_v_s), zero_division=0)
            if vf1 > best_f1:
                best_f1, best_clf = vf1, clf
        clf = best_clf

    elif probe_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        clf.fit(X_tr_s, y_tr)

    else:
        raise ValueError(f"Unknown probe_type: {probe_type!r}")

    y_pred  = clf.predict(X_te_s)
    y_proba = clf.predict_proba(X_te_s)[:, 1]

    return {
        "f1":    float(f1_score(y_te, y_pred, zero_division=0)),
        "auc":   float(roc_auc_score(y_te, y_proba)),
        "auprc": float(average_precision_score(y_te, y_proba)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_layer_curves(rows: list[dict], model: str, out_path: Path) -> None:
    if not HAS_MPL:
        return
    sub = [r for r in rows if r["model"] == model]
    if not sub:
        return
    df_sub = pd.DataFrame(sub)
    layers = sorted(df_sub["layer"].unique())
    if len(layers) < 2:
        return

    x = np.arange(len(layers))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for probe in df_sub["probe"].unique():
        ser = df_sub[df_sub["probe"] == probe].set_index("layer").reindex(layers)
        for ax, metric in zip(axes, ["f1", "auc", "auprc"]):
            ax.plot(x, ser[metric].values, marker="o", label=probe)

    for ax, metric in zip(axes, ["f1", "auc", "auprc"]):
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers], rotation=45, ha="right")
        ax.set_xlabel("Layer index")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Layer-wise {metric.upper()}")
        ax.legend()

    fig.suptitle(f"{model} — Regulatory vs Gene-Body Classification Probes")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_model_comparison(rows: list[dict], out_path: Path) -> None:
    """Bar chart of best-layer AUPRC per (model, probe)."""
    if not HAS_MPL or not rows:
        return
    df_all = pd.DataFrame(rows)
    best = (
        df_all.sort_values("auprc", ascending=False)
              .groupby(["model", "probe"])
              .first()
              .reset_index()
    )
    models = list(best["model"].unique())
    probes = list(best["probe"].unique())
    x      = np.arange(len(models))
    width  = 0.8 / max(len(probes), 1)

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(models)), 5))
    for i, probe in enumerate(probes):
        sub = best[best["probe"] == probe].set_index("model").reindex(models)
        ax.bar(x + i * width, sub["auprc"].fillna(0).values,
               width=width, label=probe)

    ax.set_xticks(x + width * (len(probes) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Best-layer AUPRC")
    ax.set_title("Regulatory vs. Gene Body — Best-layer Probe AUPRC by Model")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_root", default="data/regulatory/embeddings",
        help="Root directory with one subdirectory per model.",
    )
    parser.add_argument(
        "--windows", default="data/regulatory/windows.tsv",
        help="windows.tsv produced by 01_build_windows.py.",
    )
    parser.add_argument("--out_dir", default="data/regulatory/results")
    parser.add_argument(
        "--models", default="all",
        help="Comma-separated model names, or 'all'.",
    )
    parser.add_argument(
        "--layers", default="all",
        help="Comma-separated zero-based layer indices, or 'all'.",
    )
    parser.add_argument(
        "--skip_mlp", action="store_true",
        help="Only run logistic regression (faster).",
    )
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_root = Path(args.embeddings_root)

    all_models = discover_models(emb_root)
    if not all_models:
        raise FileNotFoundError(
            f"No model embedding directories found in {emb_root}.\n"
            f"Run 02_extract_embeddings.py first."
        )

    if args.models.strip().lower() == "all":
        models_to_run = all_models
    else:
        req = [m.strip() for m in args.models.split(",") if m.strip()]
        models_to_run = {m: all_models[m] for m in req if m in all_models}
        missing = [m for m in req if m not in all_models]
        if missing:
            print(f"WARNING: models not found in embeddings dir: {missing}")

    print(f"Available models : {list(all_models.keys())}")
    print(f"Running probes on: {list(models_to_run.keys())}")

    df = pd.read_csv(args.windows, sep="\t")
    n_tr = (df["split"] == "train").sum()
    n_v  = (df["split"] == "val").sum()
    n_te = (df["split"] == "test").sum()
    print(f"Windows: {len(df):,}  (train={n_tr:,} / val={n_v:,} / test={n_te:,})")
    print(f"Labels : {df['label'].value_counts().to_dict()}")

    probe_types = ["logistic"] if args.skip_mlp else ["logistic", "mlp"]
    rows: list[dict] = []

    for model_name, layer_files in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  ({len(layer_files)} layers available)")

        layer_indices = resolve_layer_indices(args.layers, layer_files)
        layer_indices = [l for l in layer_indices if l in layer_files]
        if not layer_indices:
            print(f"  No matching layers found, skipping.")
            continue

        for layer_idx in layer_indices:
            layer_path = layer_files[layer_idx]
            print(f"  Layer {layer_idx:02d} ({layer_path.name})")

            X_tr, y_tr = load_split(df, layer_path, "train")
            X_v,  y_v  = load_split(df, layer_path, "val")
            X_te, y_te = load_split(df, layer_path, "test")

            if not np.isfinite(X_tr).all():
                n_bad = (~np.isfinite(X_tr)).sum()
                print(f"    WARNING: {n_bad} non-finite values in train split — "
                      f"skipping layer {layer_idx}.")
                continue

            for probe in probe_types:
                metrics = train_probe(X_tr, y_tr, X_v, y_v, X_te, y_te, probe)
                print(f"    [{probe:8s}]  F1={metrics['f1']:.3f}  "
                      f"AUC={metrics['auc']:.3f}  AUPRC={metrics['auprc']:.3f}")
                rows.append({
                    "model":  model_name,
                    "layer":  layer_idx,
                    "probe":  probe,
                    **metrics,
                })

        if HAS_MPL and len(layer_indices) >= 2:
            plot_layer_curves(
                rows, model_name,
                out_dir / f"layer_curves_{model_name}.png",
            )

    results_df   = pd.DataFrame(rows)
    results_path = out_dir / "probe_results.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)
    print(f"\nResults saved to {results_path}")

    if len(models_to_run) > 1:
        plot_model_comparison(rows, out_dir / "model_comparison.png")

    # Summary
    print("\n=== Summary (best layer per model, logistic probe, AUPRC) ===")
    for model_name in models_to_run:
        sub = results_df[
            (results_df["model"] == model_name) &
            (results_df["probe"] == "logistic")
        ]
        if sub.empty:
            continue
        best = sub.sort_values("auprc").iloc[-1]
        print(f"  {model_name:12s}  layer={int(best['layer']):02d}  "
              f"F1={best['f1']:.3f}  AUC={best['auc']:.3f}  AUPRC={best['auprc']:.3f}")


if __name__ == "__main__":
    main()
