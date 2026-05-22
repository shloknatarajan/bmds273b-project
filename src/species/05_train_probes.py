"""
05_train_probes.py
------------------
Trains layer-wise probes on the cached Evo 2 embeddings for two tasks:

Task A — Species classification
  Input:  mean-pooled embedding from one layer
  Output: species label (multi-class)
  Probes: L2 logistic regression + shallow MLP
  Metrics: AUPRC (macro OvR), macro AUC, macro F1

Task B — Phylogenetic distance correlation
  Input:  all pairwise cosine / Euclidean distances between
          test-split fragment embeddings
  Output: Spearman rho vs GTDB patristic distance
  (no training needed — purely geometric evaluation)

By default we probe **every** layer that was extracted in step 04, so
layer-wise curves are produced. The Evo 2 paper highlights layer 26 as
the most informative single layer; pass ``--probe_layers 26`` (or any
comma-separated list) to restrict the run.

Outputs
-------
processed_data/results/
    classification_results.tsv  — (layer, probe, metric, value)
    phylo_correlation.tsv       — (layer, distance_type, spearman_rho, p_value)
    layer_curves.png            — layer-wise probe performance curves
    phylo_curves.png            — layer-wise Spearman rho curves

Usage
-----
  python 05_train_probes.py \
      --embeddings_dir processed_data/embeddings/evo2 \
      --fragments      processed_data/fragments.tsv \
      --phylo_dir      processed_data/phylo \
      --out_dir        processed_data/results
"""

import argparse
import re
from pathlib import Path

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
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


LAYER_FILE_RE = re.compile(r"layer_(\d+)\.h5$")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def discover_layer_files(emb_dir: Path) -> dict[int, Path]:
    """
    Map layer index -> HDF5 path for every layer file present in emb_dir.
    Tolerant of gaps (e.g. only ``layer_26.h5`` exists).
    """
    out: dict[int, Path] = {}
    for p in emb_dir.glob("layer_*.h5"):
        m = LAYER_FILE_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def load_split(
    df: pd.DataFrame,
    layer_path: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (X, species_labels, frag_ids) for a given split.
    """
    mask = df["split"] == split
    row_indices = np.where(mask)[0]

    with h5py.File(layer_path, "r") as fh:
        X = fh["embeddings"][row_indices, :]   # (N_split, D)

    species = df.loc[mask, "species"].values
    frag_ids = df.loc[mask, "frag_id"].tolist()
    return X, species, frag_ids


def load_phylo_distances(phylo_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the precomputed phylogenetic distance matrix from 03_build_phylo_distance.py.
    Returns (matrix, accessions_array).
    """
    data = np.load(phylo_dir / "distance_matrix.npz", allow_pickle=True)
    return data["distances"], data["accessions"]


# ---------------------------------------------------------------------------
# Task A: Classification probes
# ---------------------------------------------------------------------------

def train_classification_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    probe_type: str = "logistic",
) -> dict:
    """
    Trains a probe and evaluates on the test split.
    Returns a dict of metric -> value.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    if probe_type == "logistic":
        best_f1, best_C, best_model = -1, 1.0, None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(
                C=C, max_iter=1000,
                solver="lbfgs", n_jobs=-1,
            )
            clf.fit(X_train_s, y_train)
            val_preds = clf.predict(X_val_s)
            val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            if val_f1 > best_f1:
                best_f1, best_C, best_model = val_f1, C, clf
        clf = best_model

    elif probe_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-3,          # L2 regularization
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        clf.fit(X_train_s, y_train)

    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    y_pred = clf.predict(X_test_s)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    classes = clf.classes_
    try:
        y_proba = clf.predict_proba(X_test_s)
        auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro",
            labels=classes,
        )
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=classes)
        auprc = average_precision_score(y_bin, y_proba, average="macro")
    except Exception:
        auc   = float("nan")
        auprc = float("nan")

    return {"f1": f1, "auc": auc, "auprc": auprc}


# ---------------------------------------------------------------------------
# Task B: Phylogenetic distance correlation
# ---------------------------------------------------------------------------

def phylo_correlation_for_layer(
    X_test:      np.ndarray,
    test_accessions: np.ndarray,
    phylo_mat:   np.ndarray,
    phylo_accs:  np.ndarray,
) -> dict:
    """
    Computes Spearman rho between pairwise embedding distances
    and pairwise phylogenetic distances for held-out species.
    """
    acc_to_idx = {a: i for i, a in enumerate(phylo_accs)}
    valid_mask = np.array([a in acc_to_idx for a in test_accessions])

    X_valid    = X_test[valid_mask]
    accs_valid = test_accessions[valid_mask]

    if len(X_valid) < 2:
        return {"cosine_rho": np.nan, "euclidean_rho": np.nan,
                "cosine_p": np.nan,   "euclidean_p": np.nan}

    cos_dists = cdist(X_valid, X_valid, metric="cosine")
    euc_dists = cdist(X_valid, X_valid, metric="euclidean")

    phylo_rows = np.array([acc_to_idx[a] for a in accs_valid])
    phylo_sub  = phylo_mat[np.ix_(phylo_rows, phylo_rows)]

    triu_idx = np.triu_indices(len(X_valid), k=1)
    emb_cos   = cos_dists[triu_idx]
    emb_euc   = euc_dists[triu_idx]
    phy_dist  = phylo_sub[triu_idx]

    valid_pairs = ~np.isnan(phy_dist)
    emb_cos  = emb_cos[valid_pairs]
    emb_euc  = emb_euc[valid_pairs]
    phy_dist = phy_dist[valid_pairs]

    if len(phy_dist) < 10:
        return {"cosine_rho": np.nan, "euclidean_rho": np.nan,
                "cosine_p": np.nan,   "euclidean_p": np.nan}

    rho_cos, p_cos = stats.spearmanr(emb_cos,  phy_dist)
    rho_euc, p_euc = stats.spearmanr(emb_euc, phy_dist)

    return {
        "cosine_rho":     float(rho_cos),
        "euclidean_rho":  float(rho_euc),
        "cosine_p":       float(p_cos),
        "euclidean_p":    float(p_euc),
        "n_pairs":        int(valid_pairs.sum()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_layer_curves(clf_results: pd.DataFrame, out_path: Path):
    if not HAS_MPL or clf_results.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for probe in clf_results["probe"].unique():
        sub = clf_results[clf_results["probe"] == probe]
        for ax, metric in zip(axes, ["f1", "auc", "auprc"]):
            ax.plot(sub["layer"], sub[metric], marker="o", label=probe)
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"Layer-wise {metric.upper()}")
            ax.legend()
    fig.suptitle("Evo 2 7B — Species Classification Probes")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_phylo_curves(phylo_results: pd.DataFrame, out_path: Path):
    if not HAS_MPL or phylo_results.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for dist_type, color in [("cosine", "steelblue"), ("euclidean", "darkorange")]:
        col = f"{dist_type}_rho"
        ax.plot(phylo_results["layer"], phylo_results[col],
                marker="o", label=dist_type, color=color)
    ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
    ax.axhline(0.3, linestyle=":", color="green", linewidth=0.8, label="rho=0.3 target")
    ax.axhline(0.6, linestyle=":", color="darkgreen", linewidth=0.8, label="rho=0.6 ambitious")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Evo 2 7B — Phylogenetic Distance Correlation by Layer")
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
    parser.add_argument("--embeddings_dir", default="processed_data/embeddings/evo2")
    parser.add_argument("--fragments",      default="processed_data/fragments.tsv")
    parser.add_argument("--phylo_dir",      default="processed_data/phylo")
    parser.add_argument("--out_dir",        default="processed_data/results")
    parser.add_argument("--probe_layers",   default="all",
                        help="Comma-separated layer indices, or 'all' "
                             "(default). The Evo 2 paper highlights "
                             "layer 26 as the most informative single layer.")
    parser.add_argument("--skip_mlp",       action="store_true",
                        help="Only run logistic regression (faster).")
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

    if args.probe_layers.strip().lower() == "all":
        layer_indices = sorted(available.keys())
    else:
        requested = [int(x) for x in args.probe_layers.split(",") if x.strip()]
        missing = [l for l in requested if l not in available]
        if missing:
            raise FileNotFoundError(
                f"Requested layers {missing} not present in {emb_dir}. "
                f"Available: {sorted(available.keys())}. "
                f"Re-run 04_extract_embeddings_evo2.py with --layers including them."
            )
        layer_indices = requested

    df = pd.read_csv(args.fragments, sep="\t")

    le = LabelEncoder()
    le.fit(df["species"])

    phylo_mat, phylo_accs = load_phylo_distances(Path(args.phylo_dir))
    print(f"Phylo matrix: {phylo_mat.shape}, {len(phylo_accs)} accessions")

    test_mask = df["split"] == "test"
    test_accessions = df.loc[test_mask, "accession"].values

    clf_rows   = []
    phylo_rows = []
    probe_types = ["logistic"] if args.skip_mlp else ["logistic", "mlp"]

    for layer_idx in layer_indices:
        print(f"\n=== Layer {layer_idx} ===")
        layer_path = available[layer_idx]

        X_train, y_train_raw, _ = load_split(df, layer_path, "train")
        X_val,   y_val_raw,   _ = load_split(df, layer_path, "val")
        X_test,  y_test_raw,  _ = load_split(df, layer_path, "test")

        y_train = le.transform(y_train_raw)
        y_val   = le.transform(y_val_raw)
        y_test  = le.transform(y_test_raw)

        for probe in probe_types:
            print(f"  Training {probe} probe ...")
            metrics = train_classification_probe(
                X_train, y_train, X_val, y_val, X_test, y_test, probe
            )
            print(f"    F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  AUPRC={metrics['auprc']:.3f}")
            clf_rows.append({"layer": layer_idx, "probe": probe, **metrics})

        print("  Computing phylo correlation ...")
        phylo_res = phylo_correlation_for_layer(
            X_test, test_accessions, phylo_mat, phylo_accs
        )
        print(f"    Cosine rho={phylo_res['cosine_rho']:.3f}  "
              f"Euclidean rho={phylo_res['euclidean_rho']:.3f}  "
              f"(n_pairs={phylo_res.get('n_pairs', '?')})")
        phylo_rows.append({"layer": layer_idx, **phylo_res})

    clf_df   = pd.DataFrame(clf_rows)
    phylo_df = pd.DataFrame(phylo_rows)

    clf_path   = out_dir / "classification_results.tsv"
    phylo_path = out_dir / "phylo_correlation.tsv"
    clf_df.to_csv(clf_path,   sep="\t", index=False)
    phylo_df.to_csv(phylo_path, sep="\t", index=False)
    print(f"\nResults saved to:\n  {clf_path}\n  {phylo_path}")

    plot_layer_curves(clf_df, out_dir / "layer_curves.png")
    plot_phylo_curves(phylo_df, out_dir / "phylo_curves.png")

    if not clf_df.empty:
        log_rows = clf_df[clf_df["probe"] == "logistic"]
        if not log_rows.empty:
            best_row = log_rows.sort_values("f1").iloc[-1]
            print("\n=== Summary ===")
            print(f"Best classification layer (logistic, F1): "
                  f"layer {int(best_row['layer'])} -> F1={best_row['f1']:.3f}")
    if not phylo_df.empty:
        best_phy = phylo_df.sort_values("cosine_rho").iloc[-1]
        print(f"Best phylo correlation layer (cosine rho): "
              f"layer {int(best_phy['layer'])} -> rho={best_phy['cosine_rho']:.3f}")


if __name__ == "__main__":
    main()
