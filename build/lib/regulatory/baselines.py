"""
baselines.py
---------------
Sequence-feature baselines for the regulatory vs. gene-body task.

These provide a lower bound for interpreting probe results:
if frozen-embedding probes cannot beat k-mer TF-IDF, the model
has not learned meaningful biology beyond sequence composition.

Baselines
---------
  1. k-mer TF-IDF (k = 4, 5, 6)  + L2 logistic regression
  2. GC content + dinucleotide frequencies  + L2 logistic regression

Same train / val / test chromosome split as the embedding probes.
Same metrics: F1, AUC, AUPRC (binary: gene_body=0, regulatory=1).

Outputs
-------
  data/regulatory/results/
    baseline_results.tsv    — (baseline, k, f1, auc, auprc)
    baseline_comparison.png — grouped bar chart

Usage
-----
  python src/regulatory/05_baselines.py
  python src/regulatory/05_baselines.py --ks 4,6 --windows data/regulatory/windows.tsv
"""

import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


NUCLEOTIDES = "ACGT"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def seqs_to_kmer_strings(seqs: list[str], k: int) -> list[str]:
    """Converts each sequence to a space-separated string of k-mers."""
    return [" ".join(s[i:i+k] for i in range(len(s) - k + 1)) for s in seqs]


def build_tfidf(
    seqs_tr: list[str],
    seqs_v:  list[str],
    seqs_te: list[str],
    k: int,
):
    """Fits TF-IDF on training k-mers and transforms all three splits."""
    kmer_tr = seqs_to_kmer_strings(seqs_tr, k)
    kmer_v  = seqs_to_kmer_strings(seqs_v,  k)
    kmer_te = seqs_to_kmer_strings(seqs_te, k)

    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[ACGTN]+",
        min_df=2,
        sublinear_tf=True,
    )
    X_tr = vec.fit_transform(kmer_tr)
    X_v  = vec.transform(kmer_v)
    X_te = vec.transform(kmer_te)
    return X_tr, X_v, X_te


def gc_dinuc_matrix(seqs: list[str]) -> np.ndarray:
    """
    Feature vector per sequence:
      [gc_fraction, di_AA, di_AC, ..., di_TT]  (1 + 16 = 17 dimensions)
    """
    dinucs = [a + b for a, b in itertools.product(NUCLEOTIDES, repeat=2)]
    rows = []
    for seq in seqs:
        n  = len(seq)
        gc = (seq.count("G") + seq.count("C")) / max(n, 1)
        cnt = Counter(seq[i:i+2] for i in range(n - 1))
        feats = [gc] + [cnt.get(d, 0) / max(n - 1, 1) for d in dinucs]
        rows.append(feats)
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Probe (logistic regression with C sweep on val)
# ---------------------------------------------------------------------------

def fit_logistic(X_tr, y_tr, X_v, y_v, X_te, y_te, dense=False) -> dict:
    if dense:
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_v  = sc.transform(X_v)
        X_te = sc.transform(X_te)

    best_f1, best_clf = -1.0, None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", n_jobs=-1)
        clf.fit(X_tr, y_tr)
        vf1 = f1_score(y_v, clf.predict(X_v), zero_division=0)
        if vf1 > best_f1:
            best_f1, best_clf = vf1, clf

    y_pred  = best_clf.predict(X_te)
    y_proba = best_clf.predict_proba(X_te)[:, 1]
    return {
        "f1":    float(f1_score(y_te, y_pred, zero_division=0)),
        "auc":   float(roc_auc_score(y_te, y_proba)),
        "auprc": float(average_precision_score(y_te, y_proba)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(rows: list[dict], out_path: Path) -> None:
    if not HAS_MPL or not rows:
        return
    df      = pd.DataFrame(rows)
    labels  = df["label"].tolist()
    x       = np.arange(len(labels))
    metrics = ["f1", "auc", "auprc"]
    width   = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(labels)), 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric].fillna(0).values,
               width=width, label=metric.upper())

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (test split)")
    ax.set_title("Regulatory vs. Gene Body — Sequence Feature Baselines")
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
    parser.add_argument("--windows", default="data/regulatory/windows.tsv")
    parser.add_argument("--out_dir", default="data/regulatory/results")
    parser.add_argument(
        "--ks", default="4,5,6",
        help="Comma-separated k values for k-mer TF-IDF baselines.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.windows, sep="\t")
    n_tr = (df["split"] == "train").sum()
    n_v  = (df["split"] == "val").sum()
    n_te = (df["split"] == "test").sum()
    print(f"Windows: {len(df):,}  "
          f"(train={n_tr:,} / val={n_v:,} / test={n_te:,})")
    print(f"Labels : {df['label'].value_counts().to_dict()}")

    y   = (df["label"].values == "regulatory").astype(int)
    tr  = (df["split"] == "train").values
    v   = (df["split"] == "val").values
    te  = (df["split"] == "test").values
    seqs_tr = [df["seq"].iloc[i] for i in np.where(tr)[0]]
    seqs_v  = [df["seq"].iloc[i] for i in np.where(v)[0]]
    seqs_te = [df["seq"].iloc[i] for i in np.where(te)[0]]

    ks   = [int(k.strip()) for k in args.ks.split(",")]
    rows = []

    # ---- k-mer TF-IDF ----
    for k in ks:
        print(f"\nk-mer TF-IDF (k={k}) ...")
        X_tr, X_v, X_te = build_tfidf(seqs_tr, seqs_v, seqs_te, k)
        metrics = fit_logistic(X_tr, y[tr], X_v, y[v], X_te, y[te])
        print(f"  F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  "
              f"AUPRC={metrics['auprc']:.3f}")
        rows.append({
            "label":    f"{k}-mer TF-IDF",
            "baseline": "kmer_tfidf",
            "k":        k,
            **metrics,
        })

    # ---- GC + dinucleotide ----
    print("\nGC + dinucleotide frequencies ...")
    X_tr_gc = gc_dinuc_matrix(seqs_tr)
    X_v_gc  = gc_dinuc_matrix(seqs_v)
    X_te_gc = gc_dinuc_matrix(seqs_te)
    metrics = fit_logistic(
        X_tr_gc, y[tr], X_v_gc, y[v], X_te_gc, y[te], dense=True
    )
    print(f"  F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  "
          f"AUPRC={metrics['auprc']:.3f}")
    rows.append({
        "label":    "GC+dinuc",
        "baseline": "gc_dinuc",
        "k":        None,
        **metrics,
    })

    results_df = pd.DataFrame(rows)
    out_tsv    = out_dir / "baseline_results.tsv"
    results_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nSaved {out_tsv}")

    plot_comparison(rows, out_dir / "baseline_comparison.png")

    print("\n=== Baseline Summary ===")
    for row in rows:
        print(f"  {row['label']:20s}  F1={row['f1']:.3f}  "
              f"AUC={row['auc']:.3f}  AUPRC={row['auprc']:.3f}")


if __name__ == "__main__":
    main()
