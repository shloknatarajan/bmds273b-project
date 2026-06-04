"""
05b_baseline_sequence_features.py
----------------------------------
Sequence-only baseline to compare against the Evo 2 embedding probes in
05_train_probes.py.  No model forward pass is needed — features are computed
directly from raw nucleotide sequences in fragments.tsv.

Why this is the right baseline
-------------------------------
The embedding probes (script 05) show what the *model* has learned.  This
script answers: "how much of the signal is already in the raw sequence
statistics, with no learned representation?"  The gap between these two
numbers is the value added by Evo 2.

Feature sets (additive; use --features to choose)
--------------------------------------------------
  mononuc   4-dim mono-nucleotide frequencies (A, C, G, T)
  dinuc     16-dim di-nucleotide frequencies
  trinuc    64-dim tri-nucleotide frequencies
  gc        1-dim GC content  (subset of mononuc; kept separate for ablations)
  kmer4     256-dim 4-mer frequencies
  combined  mononuc + dinuc + trinuc  [default — matches scale of low-dim
            embedding probes without blowing up to 256 features]

The feature vectors are L1-normalised so they represent probability
distributions; StandardScaler is then applied before fitting (mirrors
script 05 exactly).

Classifiers
-----------
Logistic regression (L2, C swept on val) and shallow MLP — identical
hyper-parameters to script 05 so results are directly comparable.
Use --skip_mlp to skip the MLP.

Targets
-------
Same as script 05: species, phylum, domain (or a subset via --targets).
Same species-level split is read from fragments.tsv["split"].

Outputs
-------
processed_data/results/
    baseline_classification_results.tsv   — same schema as
                                            classification_results.tsv
    baseline_summary.png                   — bar chart mirroring
                                            single_layer_summary.png

Usage
-----
  # Default: combined k-mer features, all targets, logistic + MLP
  python 05b_baseline_sequence_features.py

  # Fast logistic-only run
  python 05b_baseline_sequence_features.py --skip_mlp

  # Ablation: how far does just GC content get you?
  python 05b_baseline_sequence_features.py --features gc --skip_mlp

  # Compare to a specific Evo 2 result file
  python 05b_baseline_sequence_features.py \\
      --evo2_results processed_data/results/classification_results.tsv
"""

import argparse
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_BASES = list("ACGT")


def _kmer_freqs(seq: str, k: int) -> np.ndarray:
    """L1-normalised k-mer frequency vector (4^k dims, lexicographic order)."""
    kmers = ["".join(p) for p in itertools.product(_BASES, repeat=k)]
    idx   = {km: i for i, km in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=np.float32)
    n = len(seq) - k + 1
    if n <= 0:
        return counts
    for i in range(n):
        km = seq[i : i + k]
        if km in idx:            # skip windows containing N
            counts[idx[km]] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def compute_features(seq: str, feature_set: str) -> np.ndarray:
    """
    Returns a 1-D float32 feature vector for one sequence.

    feature_set choices
    -------------------
    gc        : [gc_fraction]                 (1-D)
    mononuc   : A/C/G/T frequencies           (4-D)
    dinuc     : di-nucleotide frequencies     (16-D)
    trinuc    : tri-nucleotide frequencies    (64-D)
    kmer4     : 4-mer frequencies             (256-D)
    combined  : mononuc + dinuc + trinuc      (84-D)   [default]
    """
    seq = seq.upper()

    if feature_set == "gc":
        gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
        return np.array([gc], dtype=np.float32)

    if feature_set == "mononuc":
        return _kmer_freqs(seq, 1)

    if feature_set == "dinuc":
        return _kmer_freqs(seq, 2)

    if feature_set == "trinuc":
        return _kmer_freqs(seq, 3)

    if feature_set == "kmer4":
        return _kmer_freqs(seq, 4)

    if feature_set == "combined":
        return np.concatenate([
            _kmer_freqs(seq, 1),   # 4-D
            _kmer_freqs(seq, 2),   # 16-D
            _kmer_freqs(seq, 3),   # 64-D
        ])

    raise ValueError(f"Unknown feature_set: {feature_set!r}. "
                     f"Choose from: gc, mononuc, dinuc, trinuc, kmer4, combined")


def build_feature_matrix(sequences: "list[str]", feature_set: str) -> np.ndarray:
    print(f"  Computing '{feature_set}' features for {len(sequences)} sequences ...")
    X = np.vstack([compute_features(s, feature_set) for s in sequences])
    print(f"  Feature matrix shape: {X.shape}")
    return X


# ---------------------------------------------------------------------------
# Classification — mirrors train_classification_probe() in script 05 exactly
# ---------------------------------------------------------------------------

def train_classification_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    probe_type: str = "logistic",
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
        raise ValueError(f"Unknown probe_type: {probe_type!r}")

    y_pred = clf.predict(X_test_s)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

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
# Majority-class dummy baseline (sanity floor)
# ---------------------------------------------------------------------------

def majority_class_f1(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """F1 of always predicting the most common training class."""
    majority = np.bincount(y_train).argmax()
    y_pred   = np.full_like(y_test, fill_value=majority)
    return float(f1_score(y_test, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    baseline_df: pd.DataFrame,
    evo2_df: Optional[pd.DataFrame],
    out_path: Path,
):
    if not HAS_MPL or baseline_df.empty:
        return

    targets  = list(baseline_df["target"].drop_duplicates())
    metrics  = ["f1", "auc", "auprc"]
    n_cols   = len(targets)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
    axes = axes[0]

    for ax, target in zip(axes, targets):
        sub_bl = baseline_df[baseline_df["target"] == target]
        width  = 0.25
        probes = list(sub_bl["probe"].drop_duplicates())

        # Colour palette: baseline shades vs Evo 2 shades
        bl_colors  = ["#4C72B0", "#55A868"]     # blue, green
        evo_colors = ["#DD8452", "#C44E52"]      # orange, red

        x = np.arange(len(metrics))
        offset = 0
        for pi, probe in enumerate(probes):
            row = sub_bl[sub_bl["probe"] == probe].iloc[0]
            vals = [row[m] if np.isfinite(row[m]) else 0.0 for m in metrics]
            ax.bar(
                x + offset * width, vals, width=width,
                color=bl_colors[pi % len(bl_colors)],
                label=f"baseline/{probe}",
            )
            offset += 1

        if evo2_df is not None and not evo2_df.empty:
            sub_e2 = evo2_df[evo2_df["target"] == target]
            # Best layer per probe by F1
            for pi, probe in enumerate(sub_e2["probe"].drop_duplicates()):
                best = (sub_e2[sub_e2["probe"] == probe]
                        .sort_values("f1").iloc[-1])
                layer_short = best["layer"].split(".")
                layer_lbl   = ".".join(layer_short[-3:]) if len(layer_short) >= 3 \
                              else best["layer"]
                vals = [best[m] if np.isfinite(best[m]) else 0.0 for m in metrics]
                ax.bar(
                    x + offset * width, vals, width=width,
                    color=evo_colors[pi % len(evo_colors)],
                    label=f"evo2/{probe} ({layer_lbl})",
                )
                offset += 1

        ax.set_xticks(x + width * (offset - 1) / 2)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title(f"target = {target}")
        ax.legend(fontsize=7)

    fig.suptitle("Sequence baseline vs Evo 2 embeddings")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_TARGETS  = "species,phylum,domain"
DEFAULT_FEATURES = "combined"


def main():
    parser = argparse.ArgumentParser(
        description="Sequence-only baseline — compare to Evo 2 embedding probes."
    )
    parser.add_argument(
        "--fragments", default="processed_data/fragments.tsv",
        help="Path to fragments.tsv produced by 02_extract_fragments.py",
    )
    parser.add_argument(
        "--out_dir", default="processed_data/results",
        help="Where to write baseline_classification_results.tsv and plots.",
    )
    parser.add_argument(
        "--features", default=DEFAULT_FEATURES,
        choices=["gc", "mononuc", "dinuc", "trinuc", "kmer4", "combined"],
        help=(
            "Sequence feature set to use as the input representation.\n"
            "  gc       : GC fraction only (1-D) — minimum-information baseline\n"
            "  mononuc  : mono-nucleotide frequencies (4-D)\n"
            "  dinuc    : di-nucleotide frequencies (16-D)\n"
            "  trinuc   : tri-nucleotide frequencies (64-D)\n"
            "  kmer4    : 4-mer frequencies (256-D)\n"
            "  combined : mononuc+dinuc+trinuc concatenated (84-D) [default]"
        ),
    )
    parser.add_argument(
        "--targets", default=DEFAULT_TARGETS,
        help=(
            "Comma-separated list of classification targets.  Must be column "
            "names in fragments.tsv.  Default: 'species,phylum,domain'."
        ),
    )
    parser.add_argument(
        "--skip_mlp", action="store_true",
        help="Only run logistic regression (faster, matches --skip_mlp in script 05).",
    )
    parser.add_argument(
        "--evo2_results", default=None,
        help=(
            "Optional path to classification_results.tsv from script 05.  "
            "When provided, a side-by-side comparison plot is generated and "
            "a delta summary is printed."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load fragments
    # ------------------------------------------------------------------
    print(f"Loading fragments from {args.fragments} ...")
    df = pd.read_csv(args.fragments, sep="\t")
    print(f"  {len(df)} fragments, splits: {df['split'].value_counts().to_dict()}")

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = [t for t in targets if t not in df.columns]
    if unknown:
        raise ValueError(
            f"Unknown target column(s): {unknown}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # ------------------------------------------------------------------
    # Build feature matrices (once for all splits)
    # ------------------------------------------------------------------
    print(f"\nBuilding feature matrix (feature_set='{args.features}') ...")
    all_seqs = df["seq"].tolist()
    X_all = build_feature_matrix(all_seqs, args.features)

    train_mask = (df["split"] == "train").values
    val_mask   = (df["split"] == "val").values
    test_mask  = (df["split"] == "test").values

    X_train = X_all[train_mask]
    X_val   = X_all[val_mask]
    X_test  = X_all[test_mask]

    m_train = df[train_mask].reset_index(drop=True)
    m_val   = df[val_mask].reset_index(drop=True)
    m_test  = df[test_mask].reset_index(drop=True)

    print(
        f"  Split sizes  —  train: {X_train.shape[0]}, "
        f"val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Per-target class overlap diagnostic (mirrors script 05)
    # ------------------------------------------------------------------
    label_encoders: dict[str, LabelEncoder] = {}
    for target in targets:
        le = LabelEncoder()
        le.fit(df[target])
        label_encoders[target] = le
        train_cls = set(m_train[target].unique())
        test_cls  = set(m_test[target].unique())
        overlap   = train_cls & test_cls
        print(
            f"  target={target}: {len(le.classes_)} total classes, "
            f"{len(train_cls)} train / {len(test_cls)} test, "
            f"overlap={len(overlap)}"
            + ("  [HELD-OUT — F1 will be ~0]" if not overlap else "")
        )

    # ------------------------------------------------------------------
    # Train probes
    # ------------------------------------------------------------------
    probe_types = ["logistic"] if args.skip_mlp else ["logistic", "mlp"]
    rows = []

    for target in targets:
        le      = label_encoders[target]
        y_train = le.transform(m_train[target].values)
        y_val   = le.transform(m_val[target].values)
        y_test  = le.transform(m_test[target].values)

        # Majority-class sanity floor
        mc_f1 = majority_class_f1(y_train, y_test)
        print(f"\n[{target}] majority-class F1 floor = {mc_f1:.3f}")

        for probe in probe_types:
            print(f"  Training {probe} probe ...")
            metrics = train_classification_probe(
                X_train, y_train,
                X_val,   y_val,
                X_test,  y_test,
                probe_type=probe,
            )
            print(
                f"    F1={metrics['f1']:.3f}  "
                f"AUC={metrics['auc']:.3f}  "
                f"AUPRC={metrics['auprc']:.3f}"
            )
            rows.append({
                "feature_set": args.features,
                "probe":       probe,
                "target":      target,
                "majority_f1": mc_f1,
                **metrics,
            })

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    result_df = pd.DataFrame(rows)
    out_tsv   = out_dir / "baseline_classification_results.tsv"
    result_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nResults saved → {out_tsv}")

    # ------------------------------------------------------------------
    # Optional: load Evo 2 results and print delta summary
    # ------------------------------------------------------------------
    evo2_df = None
    if args.evo2_results:
        evo2_path = Path(args.evo2_results)
        if evo2_path.exists():
            evo2_df = pd.read_csv(evo2_path, sep="\t")
            print(f"\nLoaded Evo 2 results from {evo2_path}")
        else:
            print(f"WARNING: --evo2_results path not found: {evo2_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n=== Baseline Summary ===")
    print(f"{'target':<12} {'probe':<10} {'majority_F1':>11} "
          f"{'baseline_F1':>11} {'baseline_AUC':>12} {'baseline_AUPRC':>14}")
    print("-" * 65)
    for _, row in result_df.iterrows():
        print(
            f"{row['target']:<12} {row['probe']:<10} "
            f"{row['majority_f1']:>11.3f} "
            f"{row['f1']:>11.3f} "
            f"{row['auc']:>12.3f} "
            f"{row['auprc']:>14.3f}"
        )

    if evo2_df is not None:
        print("\n=== Delta: Evo 2 best layer  vs  sequence baseline ===")
        print(f"  (positive = Evo 2 is better)\n")
        print(f"{'target':<12} {'probe':<10} {'best_evo2_layer':<28} "
              f"{'Δ F1':>8} {'Δ AUC':>8} {'Δ AUPRC':>10}")
        print("-" * 80)
        for _, bl_row in result_df.iterrows():
            target = bl_row["target"]
            probe  = bl_row["probe"]
            sub    = evo2_df[(evo2_df["target"] == target) &
                             (evo2_df["probe"]  == probe)]
            if sub.empty:
                continue
            best = sub.sort_values("f1").iloc[-1]
            df1    = best["f1"]    - bl_row["f1"]
            dauc   = best["auc"]   - bl_row["auc"]
            dauprc = best["auprc"] - bl_row["auprc"]
            print(
                f"{target:<12} {probe:<10} {best['layer']:<28} "
                f"{df1:>+8.3f} {dauc:>+8.3f} {dauprc:>+10.3f}"
            )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_comparison(
        result_df,
        evo2_df,
        out_dir / "baseline_summary.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
