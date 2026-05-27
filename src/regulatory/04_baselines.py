"""
04_baselines.py
---------------
Non-LM baselines for Task 1 (gene_body vs regulatory), per
technical_steps.md step 6. These contextualize the frozen-embedding
probe numbers from 03_train_probes.py — a strong LM probe should beat
a k-mer bag-of-words and a small end-to-end CNN.

Baselines
---------
  1. k-mer TF-IDF + L2 logistic regression on raw sequence.
  2. Shallow 1D CNN on one-hot sequence, trained end-to-end.
     (requires torch; skipped automatically if torch is unavailable.)

Both consume windows.tsv directly (no embeddings needed), use the same
chromosome-level split, and report binary F1 / ROC-AUC / AUPRC with
regulatory as the positive class.

Outputs
-------
  <out_dir>/baseline_results.tsv  (method, f1, auc, auprc, n_train, n_test)

Usage
-----
  python 04_baselines.py                      # both baselines, k=5
  python 04_baselines.py --kmer_k 6 --skip_cnn
  python 04_baselines.py --cnn_epochs 5 --max_train 4000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

POS_LABEL = "regulatory"


def _metrics(y_true, y_pred, scores) -> dict:
    f1 = f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)
    y_bin = (np.asarray(y_true) == POS_LABEL).astype(int)
    auc = auprc = float("nan")
    if len(np.unique(y_bin)) > 1:
        auc = roc_auc_score(y_bin, scores)
        auprc = average_precision_score(y_bin, scores)
    return {"f1": float(f1), "auc": float(auc), "auprc": float(auprc)}


# ---------------------------------------------------------------------------
# Baseline 1: k-mer TF-IDF + logistic
# ---------------------------------------------------------------------------

def make_kmer_analyzer(k: int):
    """Return an analyzer that splits a sequence into overlapping k-mers."""
    def analyzer(seq: str):
        s = seq.upper()
        return [s[i:i + k] for i in range(len(s) - k + 1) if "N" not in s[i:i + k]]
    return analyzer


def run_kmer_baseline(df: pd.DataFrame, k: int) -> dict:
    print(f"\n[k-mer TF-IDF k={k}] vectorizing ...")
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]

    vec = TfidfVectorizer(analyzer=make_kmer_analyzer(k), lowercase=False, dtype=np.float32)
    X_tr = vec.fit_transform(tr["seq"])
    X_va = vec.transform(va["seq"])
    X_te = vec.transform(te["seq"])
    print(f"  vocab={len(vec.vocabulary_)}  X_train={X_tr.shape}")

    best_f1, clf = -1.0, None
    for C in [0.1, 1.0, 10.0]:
        m = LogisticRegression(C=C, max_iter=2000, n_jobs=-1)
        m.fit(X_tr, tr["label"].values)
        vf1 = f1_score(va["label"].values, m.predict(X_va),
                       pos_label=POS_LABEL, zero_division=0)
        if vf1 > best_f1:
            best_f1, clf = vf1, m

    pos_col = list(clf.classes_).index(POS_LABEL)
    scores = clf.predict_proba(X_te)[:, pos_col]
    out = _metrics(te["label"].values, clf.predict(X_te), scores)
    out.update(method=f"kmer_tfidf_k{k}+logreg", n_train=len(tr), n_test=len(te))
    print(f"  F1={out['f1']:.3f} AUC={out['auc']:.3f} AUPRC={out['auprc']:.3f}")
    return out


# ---------------------------------------------------------------------------
# Baseline 2: shallow 1D CNN on one-hot
# ---------------------------------------------------------------------------

def run_cnn_baseline(df: pd.DataFrame, epochs: int, batch_size: int,
                     max_train: int, seed: int) -> dict | None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("\n[CNN] torch not available — skipping. Use --skip_cnn to silence.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1D CNN] device={device}, epochs={epochs}")
    torch.manual_seed(seed)

    # A,C,G,T -> 0..3; anything else (N) -> 4 (zeroed in one-hot)
    base_to_idx = {b: i for i, b in enumerate("ACGT")}

    def encode(seq: str) -> np.ndarray:
        return np.fromiter((base_to_idx.get(c, 4) for c in seq.upper()),
                           dtype=np.int8, count=len(seq))

    class SeqDS(Dataset):
        def __init__(self, frame):
            self.idx = [encode(s) for s in frame["seq"]]
            self.y = (frame["label"].values == POS_LABEL).astype(np.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return torch.from_numpy(self.idx[i].astype(np.int64)), self.y[i]

    def collate(batch):
        idxs, ys = zip(*batch)
        ids = torch.stack(idxs)                       # (B, L)
        oh = torch.zeros(ids.size(0), ids.size(1), 5)  # (B, L, 5)
        oh.scatter_(2, ids.unsqueeze(-1), 1.0)
        oh = oh[:, :, :4].transpose(1, 2)              # drop N channel -> (B, 4, L)
        return oh, torch.tensor(ys)

    class ShallowCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=15, padding=7), nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(64, 64, kernel_size=9, padding=4), nn.ReLU(),
                nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                nn.Dropout(0.3), nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    rng = np.random.default_rng(seed)
    tr = df[df["split"] == "train"]
    if max_train and len(tr) > max_train:
        tr = tr.iloc[rng.choice(len(tr), max_train, replace=False)]
    te = df[df["split"] == "test"]

    dl_tr = DataLoader(SeqDS(tr), batch_size=batch_size, shuffle=True, collate_fn=collate)
    dl_te = DataLoader(SeqDS(te), batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = ShallowCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        print(f"  epoch {ep+1}/{epochs}  train_loss={tot/len(tr):.4f}")

    model.eval()
    scores, ys = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            scores.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
            ys.append(yb.numpy())
    scores = np.concatenate(scores)
    y_true = np.where(np.concatenate(ys) == 1, POS_LABEL, "gene_body")
    y_pred = np.where(scores >= 0.5, POS_LABEL, "gene_body")

    out = _metrics(y_true, y_pred, scores)
    out.update(method="shallow_cnn", n_train=len(tr), n_test=len(te))
    print(f"  F1={out['f1']:.3f} AUC={out['auc']:.3f} AUPRC={out['auprc']:.3f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", default="data/regulatory/windows.tsv")
    p.add_argument("--out_dir", default="data/regulatory/results")
    p.add_argument("--kmer_k", type=int, default=5)
    p.add_argument("--skip_kmer", action="store_true")
    p.add_argument("--skip_cnn", action="store_true")
    p.add_argument("--cnn_epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_train", type=int, default=0,
                   help="Subsample train windows for the CNN (0 = use all).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.windows, sep="\t")
    print(f"{len(df)} windows; split counts:\n{df['split'].value_counts().to_string()}")

    rows = []
    if not args.skip_kmer:
        rows.append(run_kmer_baseline(df, args.kmer_k))
    if not args.skip_cnn:
        res = run_cnn_baseline(df, args.cnn_epochs, args.batch_size,
                               args.max_train, args.seed)
        if res:
            rows.append(res)

    if not rows:
        print("No baselines run.")
        return
    res = pd.DataFrame(rows)[
        ["method", "f1", "auc", "auprc", "n_train", "n_test"]
    ]
    res_path = out_dir / "baseline_results.tsv"
    res.to_csv(res_path, sep="\t", index=False)
    print(f"\nSaved {res_path}\n{res.to_string(index=False)}")


if __name__ == "__main__":
    main()
