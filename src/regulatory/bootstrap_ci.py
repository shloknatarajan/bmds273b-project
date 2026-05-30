"""
Bootstrap 95% CIs on test-set AUC / AUPRC for each model's best logistic layer,
plus the GC+dinucleotide sequence baseline. The test split is small (290 windows)
so point estimates are noisy — CIs quantify whether the long-context models are
*significantly* above the short-context models and the composition baseline.

Method: train the probe once on train, get test predicted probabilities, then
resample the test set with replacement (default 2000x) and recompute the metric.

    python src/regulatory/bootstrap_ci.py
"""

import argparse
import itertools
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

NUC = "ACGT"


def load_split(df, layer_path, split):
    mask = (df["split"] == split).values
    idx = np.where(mask)[0]
    with h5py.File(layer_path, "r") as fh:
        X = fh["embeddings"][idx, :].astype(np.float32)
    y = (df.loc[mask, "label"].values == "regulatory").astype(int)
    return X, y


def gc_dinuc(seqs):
    dinucs = [a + b for a, b in itertools.product(NUC, repeat=2)]
    rows = []
    for s in seqs:
        n = len(s)
        gc = (s.count("G") + s.count("C")) / max(n, 1)
        c = Counter(s[i:i + 2] for i in range(n - 1))
        rows.append([gc] + [c.get(d, 0) / max(n - 1, 1) for d in dinucs])
    return np.asarray(rows, dtype=np.float32)


def probe_probs(X_tr, y_tr, X_te):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X_tr), y_tr)
    return clf.predict_proba(sc.transform(X_te))[:, 1]


def bootstrap(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aucs, aps = [], []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], p[s]))
        aps.append(average_precision_score(y[s], p[s]))
    q = lambda a: np.percentile(a, [2.5, 50, 97.5])
    return q(aucs), q(aps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/regulatory/windows.tsv")
    ap.add_argument("--embeddings_root", default="data/regulatory/embeddings")
    ap.add_argument("--probe_results", default="data/regulatory/results/probe_results.tsv")
    ap.add_argument("--out_dir", default="data/regulatory/results")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_csv(args.windows, sep="\t")
    pr = pd.read_csv(args.probe_results, sep="\t")
    emb_root = Path(args.embeddings_root)
    rows = []

    # best logistic layer per model
    for model in sorted(pr.model.unique()):
        sub = pr[(pr.model == model) & (pr.probe == "logistic")]
        best_layer = int(sub.sort_values("auprc").iloc[-1].layer)
        lp = emb_root / model / f"layer_{best_layer:02d}.h5"
        if not lp.exists():
            print(f"  skip {model}: {lp} missing")
            continue
        X_tr, y_tr = load_split(df, lp, "train")
        X_te, y_te = load_split(df, lp, "test")
        p = probe_probs(X_tr, y_tr, X_te)
        (al, am, ah), (pl, pm, ph) = bootstrap(y_te, p, args.n_boot)
        rows.append(dict(model=model, layer=best_layer,
                         auc_med=am, auc_lo=al, auc_hi=ah,
                         auprc_med=pm, auprc_lo=pl, auprc_hi=ph))
        print(f"{model:10s} L{best_layer:2d}  AUC {am:.3f} [{al:.3f}, {ah:.3f}]  "
              f"AUPRC {pm:.3f} [{pl:.3f}, {ph:.3f}]")

    # GC+dinuc baseline
    seqs = {sp: df.loc[df.split == sp, "seq"].tolist() for sp in ("train", "test")}
    Xtr = gc_dinuc(seqs["train"]); Xte = gc_dinuc(seqs["test"])
    ytr = (df.loc[df.split == "train", "label"].values == "regulatory").astype(int)
    yte = (df.loc[df.split == "test", "label"].values == "regulatory").astype(int)
    p = probe_probs(Xtr, ytr, Xte)
    (al, am, ah), (pl, pm, ph) = bootstrap(yte, p, args.n_boot)
    rows.append(dict(model="baseline_gc_dinuc", layer=-1,
                     auc_med=am, auc_lo=al, auc_hi=ah,
                     auprc_med=pm, auprc_lo=pl, auprc_hi=ph))
    print(f"{'baseline':10s}      AUC {am:.3f} [{al:.3f}, {ah:.3f}]  "
          f"AUPRC {pm:.3f} [{pl:.3f}, {ph:.3f}]")

    out = Path(args.out_dir) / "bootstrap_ci.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
