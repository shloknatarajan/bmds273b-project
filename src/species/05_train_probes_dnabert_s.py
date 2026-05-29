"""
05_train_probes.py
------------------
Trains probes on the cached DNABERT-S embeddings for two tasks:

Task A - Classification probes
  Targets: species (200 classes; train/val/test species are DISJOINT,
           so test F1 is ~0 by construction — kept as a "no-information
           baseline" reference) and phylum (20 classes; fully overlapping
           across splits, the real generalization metric).
  Probes:  L2 logistic regression + shallow MLP
  Metrics: macro F1, macro AUC (OvR), macro AUPRC

Task B - Phylogenetic distance correlation
  Input:  all pairwise cosine / Euclidean distances between
          test-split fragment embeddings
  Output: Spearman rho vs GTDB patristic distance
  (no training needed - purely geometric evaluation)

Layer naming
------------
Script 04 saves one HDF5 per requested layer using the safe-filename
convention (dots replaced with underscores), e.g. ``blocks.28.mlp.l3``
becomes ``blocks_28_mlp_l3.h5``. The original layer name is preserved
in the file's ``layer_name`` attribute and used as the display label.

Outputs
-------
processed_data/results/
    classification_results.tsv  - (layer, probe, target, f1, auc, auprc)
    phylo_correlation.tsv       - (layer, cosine_rho, euclidean_rho, ...)
    layer_curves_<target>.png   - probe metrics vs layer (multi-layer)
    phylo_curves.png            - Spearman rho vs layer (multi-layer)
    single_layer_summary.png    - bar chart (single-layer run)

Usage
-----
  python 05_train_probes.py                            # all cached layers
  python 05_train_probes.py --skip_mlp                 # faster (logistic only)
  python 05_train_probes.py --targets phylum           # phylum only
  python 05_train_probes.py --probe_layers blocks.28.mlp.l3
"""

import argparse
from pathlib import Path
import re
import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from typing import Optional

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Layer-file discovery
# ---------------------------------------------------------------------------

def _layer_sort_key(layer_name: str):
    """
    Sorts:
      encoder.layer.0
      layer_00
      encoder_layer_0
      pooler
    in numerical order.
    """
    if layer_name == "pooler":
        return (1_000_000, layer_name)

    patterns = [
        r"encoder\.layer\.(\d+)",
        r"encoder_layer_(\d+)",
        r"layer_(\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, layer_name)
        if m:
            return (int(m.group(1)), layer_name)

    return (999_999, layer_name)

def canonicalize_layer_name(name: str) -> str:
    """
    Convert different filename conventions to a canonical form.

    Examples:
      encoder_layer_0 -> encoder.layer.0
      layer_00        -> encoder.layer.0
    """
    if name == "pooler":
        return "pooler"

    m = re.search(r"encoder_layer_(\d+)", name)
    if m:
        return f"encoder.layer.{int(m.group(1))}"

    m = re.search(r"layer_(\d+)", name)
    if m:
        return f"encoder.layer.{int(m.group(1))}"

    return name

def discover_layer_files(emb_dir: Path) -> dict[str, Path]:
    """
    Returns:
      {
        canonical_layer_name: path_to_h5
      }

    Deduplicates aliases like:
      encoder_layer_0.h5
      layer_00.h5
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

        # keep first occurrence only
        if canonical not in found:
            found[canonical] = p

    return dict(sorted(found.items(), key=lambda kv: _layer_sort_key(kv[0])))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings_for_split(
    df: pd.DataFrame,
    layer_path: Path,
    split: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (X, metadata_df) where metadata_df has the same row order as X."""
    mask = (df["split"] == split).values
    row_indices = np.where(mask)[0]
    with h5py.File(layer_path, "r") as fh:
        X = fh["embeddings"][row_indices, :]
    meta = df.loc[mask].reset_index(drop=True)
    return X, meta


def load_phylo_distances(phylo_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(phylo_dir / "distance_matrix.npz", allow_pickle=True)
    return data["distances"], data["accessions"]


# ---------------------------------------------------------------------------
# Task A: Classification probes
# ---------------------------------------------------------------------------

def train_classification_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    probe_type: str = "logistic",
    classes_all: Optional[np.ndarray] = None,
) -> dict:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    if probe_type == "logistic":
        best_f1, best_model = -1.0, None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(
                C=C, max_iter=2000, solver="lbfgs", n_jobs=-1,
            )
            clf.fit(X_train_s, y_train)
            val_f1 = f1_score(
                y_val, clf.predict(X_val_s),
                average="macro", zero_division=0,
            )
            if val_f1 > best_f1:
                best_f1, best_model = val_f1, clf
        clf = best_model

    elif probe_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        clf.fit(X_train_s, y_train)

    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    y_pred = clf.predict(X_test_s)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # AUC / AUPRC are only meaningful when train classes cover the test
    # classes (i.e. the phylum/domain case, not the held-out-species case).
    train_classes = set(clf.classes_.tolist())
    test_classes  = set(np.unique(y_test).tolist())
    auc = float("nan")
    auprc = float("nan")
    if test_classes.issubset(train_classes):
        try:
            y_proba = clf.predict_proba(X_test_s)
            auc = roc_auc_score(
                y_test, y_proba, multi_class="ovr",
                average="macro", labels=clf.classes_,
            )
            y_bin = label_binarize(y_test, classes=clf.classes_)
            auprc = average_precision_score(y_bin, y_proba, average="macro")
        except Exception:
            pass

    return {"f1": float(f1), "auc": float(auc), "auprc": float(auprc)}


# ---------------------------------------------------------------------------
# Task B: Phylogenetic distance correlation
# ---------------------------------------------------------------------------

def phylo_correlation_for_layer(
    X_test: np.ndarray,
    test_accessions: np.ndarray,
    phylo_mat: np.ndarray,
    phylo_accs: np.ndarray,
) -> dict:
    acc_to_idx = {a: i for i, a in enumerate(phylo_accs)}
    valid_mask = np.array([a in acc_to_idx for a in test_accessions])

    X_valid    = X_test[valid_mask]
    accs_valid = test_accessions[valid_mask]

    empty = {"cosine_rho": np.nan, "euclidean_rho": np.nan,
             "cosine_p": np.nan,   "euclidean_p": np.nan, "n_pairs": 0}
    if len(X_valid) < 2:
        return empty

    cos_dists = cdist(X_valid, X_valid, metric="cosine")
    euc_dists = cdist(X_valid, X_valid, metric="euclidean")

    phylo_rows = np.array([acc_to_idx[a] for a in accs_valid])
    phylo_sub  = phylo_mat[np.ix_(phylo_rows, phylo_rows)]

    triu_idx = np.triu_indices(len(X_valid), k=1)
    emb_cos  = cos_dists[triu_idx]
    emb_euc  = euc_dists[triu_idx]
    phy_dist = phylo_sub[triu_idx]

    keep = ~np.isnan(phy_dist)
    emb_cos, emb_euc, phy_dist = emb_cos[keep], emb_euc[keep], phy_dist[keep]
    if len(phy_dist) < 10:
        return empty

    rho_cos, p_cos = stats.spearmanr(emb_cos, phy_dist)
    rho_euc, p_euc = stats.spearmanr(emb_euc, phy_dist)

    return {
        "cosine_rho":    float(rho_cos),
        "euclidean_rho": float(rho_euc),
        "cosine_p":      float(p_cos),
        "euclidean_p":   float(p_euc),
        "n_pairs":       int(keep.sum()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _layer_tick_labels(layers: list[str]) -> list[str]:
    short = []
    for name in layers:
        idx = _block_index(name)
        short.append(f"b{idx}" if idx < 10**9 else name)
    return short


def plot_layer_curves_per_target(
    clf_df: pd.DataFrame, target: str, out_path: Path,
):
    if not HAS_MPL or clf_df.empty:
        return
    sub = clf_df[clf_df["target"] == target]
    if sub.empty:
        return
    layers = list(sub["layer"].drop_duplicates())
    if len(layers) < 2:
        return  # bars cover the single-layer case
    x = np.arange(len(layers))
    ticks = _layer_tick_labels(layers)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for probe in sub["probe"].unique():
        ser = sub[sub["probe"] == probe].set_index("layer").reindex(layers)
        for ax, metric in zip(axes, ["f1", "auc", "auprc"]):
            ax.plot(x, ser[metric].values, marker="o", label=probe)
    for ax, metric in zip(axes, ["f1", "auc", "auprc"]):
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, rotation=45, ha="right")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Layer-wise {metric.upper()}")
        ax.legend()
    fig.suptitle(f"DNABERT-S - Classification Probes ({target})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_phylo_curves(phylo_df: pd.DataFrame, out_path: Path):
    if not HAS_MPL or phylo_df.empty or len(phylo_df) < 2:
        return
    layers = list(phylo_df["layer"])
    x = np.arange(len(layers))
    ticks = _layer_tick_labels(layers)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, phylo_df["cosine_rho"].values, marker="o",
            color="steelblue", label="cosine")
    ax.plot(x, phylo_df["euclidean_rho"].values, marker="o",
            color="darkorange", label="euclidean")
    ax.axhline(0,   linestyle="--", color="gray",      linewidth=0.8)
    ax.axhline(0.3, linestyle=":",  color="green",     linewidth=0.8, label="rho=0.3 target")
    ax.axhline(0.6, linestyle=":",  color="darkgreen", linewidth=0.8, label="rho=0.6 ambitious")
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_title("DNABERT-S - Phylogenetic Distance Correlation by Layer")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_single_layer_summary(
    clf_df: pd.DataFrame, phylo_df: pd.DataFrame, out_path: Path,
):
    if not HAS_MPL:
        return
    targets = list(clf_df["target"].drop_duplicates()) if not clf_df.empty else []
    n_cols = max(1, len(targets)) + (1 if not phylo_df.empty else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
    axes = axes[0]

    for ax, target in zip(axes, targets):
        sub = clf_df[clf_df["target"] == target]
        probes = list(sub["probe"])
        metrics = ["f1", "auc", "auprc"]
        width = 0.35
        positions = np.arange(len(metrics))
        for i, probe in enumerate(probes):
            row = sub[sub["probe"] == probe].iloc[0]
            vals = [row[m] if np.isfinite(row[m]) else 0.0 for m in metrics]
            ax.bar(positions + i * width, vals, width=width, label=probe)
        ax.set_xticks(positions + width * (len(probes) - 1) / 2)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1.0)
        ax.set_title(f"target={target}")
        ax.legend()

    if not phylo_df.empty:
        ax = axes[len(targets)]
        row = phylo_df.iloc[0]
        ax.bar(["cosine", "euclidean"],
               [row["cosine_rho"], row["euclidean_rho"]],
               color=["steelblue", "darkorange"])
        ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
        ax.set_ylabel("Spearman rho")
        ax.set_title("Phylo distance correlation")

    layer_name = (clf_df["layer"].iloc[0] if not clf_df.empty
                  else phylo_df["layer"].iloc[0])
    fig.suptitle(f"DNABERT-S - {layer_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_TARGETS = "species,phylum"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", default="processed_data/embeddings/dnaberts")
    parser.add_argument("--fragments",      default="processed_data/fragments.tsv")
    parser.add_argument("--phylo_dir",      default="processed_data/phylo")
    parser.add_argument("--out_dir",        default="processed_data/results")
    parser.add_argument("--probe_layers",
                        default="encoder.layer.0,encoder.layer.5,encoder.layer.9,encoder.layer.10,encoder.layer.11,pooler",
                        help="Comma-separated layer names (dotted or "
                             "underscored), or 'all' (default).")
    parser.add_argument("--targets",        default=DEFAULT_TARGETS,
                        help=f"Comma-separated classification targets. "
                             f"Default: '{DEFAULT_TARGETS}'. Valid columns "
                             f"from fragments.tsv: species, phylum, domain.")
    parser.add_argument("--skip_mlp",       action="store_true",
                        help="Only run logistic regression (faster).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.embeddings_dir)

    available = discover_layer_files(emb_dir)
    if not available:
        raise FileNotFoundError(
            f"No .h5 files found in {emb_dir}. "
            f"Run 04_extract_embeddings_dnabert_s.py first."
        )
    print(f"----------------\nFound {len(available)} layer file(s) in {emb_dir}: "
          f"{list(available.keys())}")

    if args.probe_layers.strip().lower() == "all":
        layer_names = list(available.keys())
    else:
        layer_names = [
            s.strip()
            for s in args.probe_layers.split(",")
            if s.strip()
        ]

    missing = [x for x in layer_names if x not in available]

    if missing:
        raise ValueError(
            f"Missing layers: {missing}\n"
            f"Available layers: {list(available.keys())}"
        )

    df = pd.read_csv(args.fragments, sep="\t")

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = [t for t in targets if t not in df.columns]
    if unknown:
        raise ValueError(
            f"Unknown target column(s): {unknown}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Diagnostic: report train/test class overlap per target so it's
    # obvious when a target is a held-out-class generalization task.
    label_encoders: dict[str, LabelEncoder] = {}
    for target in targets:
        le = LabelEncoder()
        le.fit(df[target])
        label_encoders[target] = le
        train_classes = set(df.loc[df["split"] == "train", target].unique())
        test_classes  = set(df.loc[df["split"] == "test",  target].unique())
        overlap = train_classes & test_classes
        print(f"  target={target}: {len(le.classes_)} total classes, "
              f"{len(train_classes)} train / {len(test_classes)} test, "
              f"overlap={len(overlap)}"
              + ("  [HELD-OUT — F1 will be ~0]" if not overlap else ""))

    phylo_mat, phylo_accs = load_phylo_distances(Path(args.phylo_dir))
    print(f"Phylo matrix: {phylo_mat.shape}, {len(phylo_accs)} accessions")

    clf_rows, phylo_rows = [], []
    probe_types = ["logistic"] if args.skip_mlp else ["logistic", "mlp"]

    for layer_name in layer_names:
        layer_path = available[layer_name]
        print(f"\n=== {layer_name} ({layer_path.name}) ===")

        X_train, m_train = load_embeddings_for_split(df, layer_path, "train")
        X_val,   m_val   = load_embeddings_for_split(df, layer_path, "val")
        X_test,  m_test  = load_embeddings_for_split(df, layer_path, "test")

        if not np.isfinite(X_train).all():
            n_bad = (~np.isfinite(X_train)).sum()
            raise ValueError(
                f"Embeddings for {layer_name} contain {n_bad} non-finite "
                f"values in the train split - re-run 04_extract_embeddings_dnabert_s.py."
            )

        for target in targets:
            le = label_encoders[target]
            y_train = le.transform(m_train[target].values)
            y_val   = le.transform(m_val[target].values)
            y_test  = le.transform(m_test[target].values)

            for probe in probe_types:
                print(f"  [{target}] training {probe} probe ...")
                metrics = train_classification_probe(
                    X_train, y_train, X_val, y_val, X_test, y_test, probe,
                )
                print(f"    F1={metrics['f1']:.3f}  "
                      f"AUC={metrics['auc']:.3f}  "
                      f"AUPRC={metrics['auprc']:.3f}")
                clf_rows.append({
                    "layer": layer_name, "probe": probe,
                    "target": target, **metrics,
                })

        print("  Computing phylo correlation ...")
        phylo_res = phylo_correlation_for_layer(
            X_test, m_test["accession"].values, phylo_mat, phylo_accs,
        )
        print(f"    cosine rho={phylo_res['cosine_rho']:.3f}  "
              f"euclidean rho={phylo_res['euclidean_rho']:.3f}  "
              f"(n_pairs={phylo_res.get('n_pairs', '?')})")
        phylo_rows.append({"layer": layer_name, **phylo_res})

    clf_df   = pd.DataFrame(clf_rows)
    phylo_df = pd.DataFrame(phylo_rows)

    clf_path   = out_dir / "classification_results.tsv"
    phylo_path = out_dir / "phylo_correlation.tsv"
    clf_df.to_csv(clf_path,   sep="\t", index=False)
    phylo_df.to_csv(phylo_path, sep="\t", index=False)
    print(f"\nResults saved to:\n  {clf_path}\n  {phylo_path}")

    if len(layer_names) >= 2:
        for target in targets:
            plot_layer_curves_per_target(
                clf_df, target, out_dir / f"layer_curves_{target}.png",
            )
        plot_phylo_curves(phylo_df, out_dir / "phylo_curves.png")
    else:
        plot_single_layer_summary(clf_df, phylo_df, out_dir / "single_layer_summary.png")

    print("\n=== Summary ===")
    for target in targets:
        sub = clf_df[(clf_df["target"] == target) & (clf_df["probe"] == "logistic")]
        if sub.empty:
            continue
        best = sub.sort_values("f1").iloc[-1]
        print(f"Best layer for {target} (logistic, F1): "
              f"{best['layer']} -> F1={best['f1']:.3f}, "
              f"AUC={best['auc']:.3f}, AUPRC={best['auprc']:.3f}")
    if not phylo_df.empty:
        best = phylo_df.sort_values("cosine_rho").iloc[-1]
        print(f"Best phylo correlation layer (cosine rho): "
              f"{best['layer']} -> rho={best['cosine_rho']:.3f}")


if __name__ == "__main__":
    main()
