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
  Output: Spearman ρ vs GTDB patristic distance
  (no training needed — purely geometric evaluation)

Outputs
-------
results/
    classification_results.tsv  — (layer, probe, metric, value)
    phylo_correlation.tsv       — (layer, distance_type, spearman_rho, p_value)
    layer_curves.png            — layer-wise probe performance curves
    phylo_curves.png            — layer-wise Spearman ρ curves

Usage
-----
  python 05_train_probes.py \
      --embeddings_dir data/embeddings/evo2 \
      --fragments      data/fragments/fragments.tsv \
      --phylo_dir      data/phylo \
      --out_dir        results
"""

import argparse
import json
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


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_split(
    df: pd.DataFrame,
    embeddings_dir: Path,
    layer_idx: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (X, y_encoded, frag_ids) for a given split and layer.
    """
    mask = df["split"] == split
    row_indices = np.where(mask)[0]

    with h5py.File(embeddings_dir / f"layer_{layer_idx:02d}.h5", "r") as fh:
        X = fh["embeddings"][row_indices, :]   # (N_split, D)

    species = df.loc[mask, "species"].values
    frag_ids = df.loc[mask, "frag_id"].tolist()
    return X, species, frag_ids


def load_phylo_distances(
    phylo_dir: Path,
    accessions_in_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the precomputed phylogenetic distance matrix.
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
    Returns a dict of metric → value.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    if probe_type == "logistic":
        # Try a few regularization strengths on val, pick best
        best_f1, best_C, best_model = -1, 1.0, None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(
                C=C, max_iter=1000, multi_class="multinomial",
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

    # Test metrics
    y_pred = clf.predict(X_test_s)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # AUC & AUPRC require probability scores
    classes = clf.classes_
    try:
        y_proba = clf.predict_proba(X_test_s)
        # One-vs-rest macro AUC
        auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro",
            labels=classes,
        )
        # Macro average precision
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
    Computes Spearman ρ between pairwise embedding distances
    and pairwise phylogenetic distances for held-out species.
    """
    # Map test accessions to rows in the phylo matrix
    acc_to_idx = {a: i for i, a in enumerate(phylo_accs)}
    valid_mask = np.array([a in acc_to_idx for a in test_accessions])

    X_valid   = X_test[valid_mask]
    accs_valid = test_accessions[valid_mask]

    if len(X_valid) < 2:
        return {"cosine_rho": np.nan, "euclidean_rho": np.nan,
                "cosine_p": np.nan,   "euclidean_p": np.nan}

    # Pairwise embedding distances (upper triangle)
    cos_dists = cdist(X_valid, X_valid, metric="cosine")
    euc_dists = cdist(X_valid, X_valid, metric="euclidean")

    # Corresponding phylogenetic distances
    phylo_rows = np.array([acc_to_idx[a] for a in accs_valid])
    phylo_sub  = phylo_mat[np.ix_(phylo_rows, phylo_rows)]

    # Upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(len(X_valid), k=1)
    emb_cos   = cos_dists[triu_idx]
    emb_euc   = euc_dists[triu_idx]
    phy_dist  = phylo_sub[triu_idx]

    # Remove NaN phylo pairs (cross-domain)
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
    if not HAS_MPL:
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
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for dist_type, color in [("cosine", "steelblue"), ("euclidean", "darkorange")]:
        col = f"{dist_type}_rho"
        ax.plot(phylo_results["layer"], phylo_results[col],
                marker="o", label=dist_type, color=color)
    ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
    ax.axhline(0.3, linestyle=":", color="green", linewidth=0.8, label="ρ=0.3 target")
    ax.axhline(0.6, linestyle=":", color="darkgreen", linewidth=0.8, label="ρ=0.6 ambitious")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ")
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
    parser.add_argument("--embeddings_dir", default="data/embeddings/evo2")
    parser.add_argument("--fragments",      default="data/fragments/fragments.tsv")
    parser.add_argument("--phylo_dir",      default="data/phylo")
    parser.add_argument("--out_dir",        default="results")
    parser.add_argument("--probe_layers",   default="all",
                        help="Comma-separated layer indices, or 'all'.")
    parser.add_argument("--skip_mlp",       action="store_true",
                        help="Only run logistic regression (faster).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.embeddings_dir)

    # Determine number of layers from HDF5 files
    layer_files = sorted(emb_dir.glob("layer_*.h5"))
    n_layers = len(layer_files)
    print(f"Found {n_layers} layer files in {emb_dir}")

    if args.probe_layers == "all":
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [int(x) for x in args.probe_layers.split(",")]

    # Load fragment metadata
    df = pd.read_csv(args.fragments, sep="\t")

    # Encode species labels
    le = LabelEncoder()
    le.fit(df["species"])

    # Load phylogenetic distances
    phylo_mat, phylo_accs = load_phylo_distances(Path(args.phylo_dir), None)
    print(f"Phylo matrix: {phylo_mat.shape}, {len(phylo_accs)} accessions")

    # Map test fragments to accessions (for phylo task)
    test_mask = df["split"] == "test"
    test_accessions = df.loc[test_mask, "accession"].values

    clf_rows   = []
    phylo_rows = []
    probe_types = ["logistic"] if args.skip_mlp else ["logistic", "mlp"]

    for layer_idx in layer_indices:
        print(f"\n=== Layer {layer_idx}/{n_layers - 1} ===")

        # --- Load embeddings for each split ---
        X_train, y_train_raw, _ = load_split(df, emb_dir, layer_idx, "train")
        X_val,   y_val_raw,   _ = load_split(df, emb_dir, layer_idx, "val")
        X_test,  y_test_raw,  _ = load_split(df, emb_dir, layer_idx, "test")

        y_train = le.transform(y_train_raw)
        y_val   = le.transform(y_val_raw)
        y_test  = le.transform(y_test_raw)

        # --- Task A: classification probes ---
        for probe in probe_types:
            print(f"  Training {probe} probe ...")
            metrics = train_classification_probe(
                X_train, y_train, X_val, y_val, X_test, y_test, probe
            )
            print(f"    F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  AUPRC={metrics['auprc']:.3f}")
            clf_rows.append({"layer": layer_idx, "probe": probe, **metrics})

        # --- Task B: phylogenetic correlation ---
        print("  Computing phylo correlation ...")
        phylo_res = phylo_correlation_for_layer(
            X_test, test_accessions, phylo_mat, phylo_accs
        )
        print(f"    Cosine ρ={phylo_res['cosine_rho']:.3f}  "
              f"Euclidean ρ={phylo_res['euclidean_rho']:.3f}  "
              f"(n_pairs={phylo_res.get('n_pairs', '?')})")
        phylo_rows.append({"layer": layer_idx, **phylo_res})

    # --- Save results ---
    clf_df   = pd.DataFrame(clf_rows)
    phylo_df = pd.DataFrame(phylo_rows)

    clf_path   = out_dir / "classification_results.tsv"
    phylo_path = out_dir / "phylo_correlation.tsv"
    clf_df.to_csv(clf_path,   sep="\t", index=False)
    phylo_df.to_csv(phylo_path, sep="\t", index=False)
    print(f"\nResults saved to:\n  {clf_path}\n  {phylo_path}")

    # --- Plot ---
    plot_layer_curves(clf_df, out_dir / "layer_curves.png")
    plot_phylo_curves(phylo_df, out_dir / "phylo_curves.png")

    # --- Summary ---
    best_row = clf_df[clf_df["probe"] == "logistic"].sort_values("f1").iloc[-1]
    best_phy = phylo_df.sort_values("cosine_rho").iloc[-1]
    print("\n=== Summary ===")
    print(f"Best classification layer (logistic, F1): "
          f"layer {int(best_row['layer'])} → F1={best_row['f1']:.3f}")
    print(f"Best phylo correlation layer (cosine ρ): "
          f"layer {int(best_phy['layer'])} → ρ={best_phy['cosine_rho']:.3f}")


if __name__ == "__main__":
    main()
