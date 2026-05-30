"""
Strengthening analyses on existing embeddings (no re-extraction):

1. PAIRED bootstrap of the AUC *difference* between conditions on the same
   resampled test windows. Comparing marginal CIs understates significance when
   predictions are correlated; the paired test directly asks "is model A > model B
   on these windows?" and gives a bootstrap p-value (fraction of resamples with
   Δ<=0).
2. LABEL-SHUFFLE null control: retrain the best HyenaDNA probe on permuted train
   labels; test AUC should collapse to chance, confirming the pipeline isn't
   leaking.

    python src/regulatory/paired_bootstrap.py
"""

import itertools
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

WIN = "data/regulatory/windows.tsv"
EMB = Path("data/regulatory/embeddings")
PR = "data/regulatory/results/probe_results.tsv"
NUC = "ACGT"


def load_split(df, layer_path, split):
    mask = (df["split"] == split).values
    idx = np.where(mask)[0]
    with h5py.File(layer_path, "r") as fh:
        X = fh["embeddings"][idx, :].astype(np.float32)
    y = (df.loc[mask, "label"].values == "regulatory").astype(int)
    return X, y


def gc_dinuc(seqs):
    dn = [a + b for a, b in itertools.product(NUC, repeat=2)]
    out = []
    for s in seqs:
        n = len(s)
        gc = (s.count("G") + s.count("C")) / max(n, 1)
        c = Counter(s[i:i + 2] for i in range(n - 1))
        out.append([gc] + [c.get(d, 0) / max(n - 1, 1) for d in dn])
    return np.asarray(out, dtype=np.float32)


def test_probs(X_tr, y_tr, X_te):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X_tr), y_tr)
    return clf.predict_proba(sc.transform(X_te))[:, 1]


def main():
    df = pd.read_csv(WIN, sep="\t")
    pr = pd.read_csv(PR, sep="\t")
    yte = (df.loc[df.split == "test", "label"].values == "regulatory").astype(int)

    # collect best-layer test probabilities per model + baseline
    probs = {}
    for m in sorted(pr.model.unique()):
        bl = int(pr[(pr.model == m) & (pr.probe == "logistic")].sort_values("auprc").iloc[-1].layer)
        lp = EMB / m / f"layer_{bl:02d}.h5"
        if not lp.exists():
            continue
        Xtr, ytr = load_split(df, lp, "train")
        Xte, _ = load_split(df, lp, "test")
        probs[m] = test_probs(Xtr, ytr, Xte)
    seqs_tr = df.loc[df.split == "train", "seq"].tolist()
    seqs_te = df.loc[df.split == "test", "seq"].tolist()
    ytr_b = (df.loc[df.split == "train", "label"].values == "regulatory").astype(int)
    probs["baseline"] = test_probs(gc_dinuc(seqs_tr), ytr_b, gc_dinuc(seqs_te))

    # paired bootstrap of AUC differences
    contrasts = [("hyenadna", "nt"), ("hyenadna", "dnabert_s"), ("hyenadna", "baseline"),
                 ("evo2", "nt"), ("evo2", "dnabert_s"), ("evo2", "baseline"),
                 ("hyenadna", "evo2")]
    rng = np.random.default_rng(0)
    idx = np.arange(len(yte))
    NB = 5000
    print("=== Paired bootstrap of ΔAUC (A − B), same resampled windows ===")
    rows = []
    for a, b in contrasts:
        if a not in probs or b not in probs:
            continue
        diffs = []
        for _ in range(NB):
            s = rng.choice(idx, len(idx), replace=True)
            if len(np.unique(yte[s])) < 2:
                continue
            diffs.append(roc_auc_score(yte[s], probs[a][s]) - roc_auc_score(yte[s], probs[b][s]))
        diffs = np.array(diffs)
        lo, med, hi = np.percentile(diffs, [2.5, 50, 97.5])
        p_le0 = float((diffs <= 0).mean())  # bootstrap p that A not better than B
        sig = "*" if p_le0 < 0.05 else " "
        print(f"  {a:9s} - {b:9s}: ΔAUC {med:+.3f} [{lo:+.3f},{hi:+.3f}]  p(Δ<=0)={p_le0:.3f} {sig}")
        rows.append(dict(a=a, b=b, dauc_med=med, dauc_lo=lo, dauc_hi=hi, p_le0=p_le0))
    pd.DataFrame(rows).to_csv("data/regulatory/results/paired_bootstrap.tsv", sep="\t", index=False)

    # label-shuffle null control on HyenaDNA best layer
    bl = int(pr[(pr.model == "hyenadna") & (pr.probe == "logistic")].sort_values("auprc").iloc[-1].layer)
    Xtr, ytr = load_split(df, EMB / "hyenadna" / f"layer_{bl:02d}.h5", "train")
    Xte, _ = load_split(df, EMB / "hyenadna" / f"layer_{bl:02d}.h5", "test")
    perm = rng.permutation(ytr)
    p_shuf = test_probs(Xtr, perm, Xte)
    print(f"\n=== Null control: HyenaDNA L{bl} with SHUFFLED train labels ===")
    print(f"  test AUC = {roc_auc_score(yte, p_shuf):.3f}  (expect ~0.5 — confirms no leakage)")


if __name__ == "__main__":
    main()
