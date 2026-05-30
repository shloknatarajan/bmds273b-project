# Next Step (optional): Scale the Regulatory Dataset to Tighten Significance

**Date:** 2026-05-30T17:27:51+0000
**Status:** NOT YET RUN — documented for a future session.
**Why:** Task 1 is complete with 4 models (see `docs/results/regulatory_paper.md`).
The headline claims are already significant (long-context > baseline: paired
bootstrap p≤0.02; HyenaDNA > NT: p=0.027), but the test split is small (n=290) so
the *secondary* long-vs-short contrasts stay at trend level (Evo 2 − NT p=0.09,
HyenaDNA − DNABERT-S p=0.07). Scaling the window count 2–3× shrinks the CIs and
would likely push these to significance. Same method, same conclusions — just
tighter. Estimated **~1–2 h**, mostly Evo 2 on A100.

---

## What limits the current dataset size

`build_windows.py` GC-matching is the bottleneck: with `--n_windows 5000
--pool_size 40000` we retained only **2,012 / class** (4,024 total; test = 290).
The matcher caps each GC bin at `min(gene_in_bin, reg_in_bin, n_windows/gc_bins)`,
so a bigger candidate **pool** (more per bin) + higher `n_windows` yields more
matched windows. Target ~8–12k total → test split ~600–900.

## Procedure (run from repo root)

```bash
cd /home/ec2-user/bmds273b-project
set -a; . ./.env 2>/dev/null; set +a

# 1. Re-build windows, larger pool + target. (Modal CPU, 32 GB, ~10 min.)
#    NOTE: this overwrites windows.tsv → ALL embeddings must be re-extracted,
#    because row order / window set changes. Consider --out_dir a NEW dir
#    (e.g. data/regulatory_v2) to keep the n=290 results for comparison.
.venv/bin/modal run src/modal_app.py::build  # edit build() args first:
#    in src/modal_app.py build(): n_windows=10000, pool_size=200000
#    (or add CLI passthrough). Verify split_stats.txt shows the larger,
#    still-balanced, GC-matched counts BEFORE extracting.

# 2. Re-extract all 4 models on the new windows.tsv.
.venv/bin/modal run src/modal_app.py --stage extract --models hyenadna,nt,dnabert_s
.venv/bin/modal run src/modal_evo2.py                      # A100, ~30–60 min

# 3. Pull embeddings locally (flatten the nested <model>/<model>/ dir each time).
for m in hyenadna nt dnabert_s evo2; do
  rm -rf data/regulatory/embeddings/$m; mkdir -p data/regulatory/embeddings/$m
  .venv/bin/modal volume get bmds273b-reg-data regulatory/embeddings/$m/ data/regulatory/embeddings/$m/ --force
  [ -d data/regulatory/embeddings/$m/$m ] && mv data/regulatory/embeddings/$m/$m/* data/regulatory/embeddings/$m/
done

# 4. Re-run analysis (CPU, regvenv). Probes write probe_results.tsv; merge evo2.
RV=/home/ec2-user/regvenv/bin/python
$RV src/regulatory/train_probes.py --models hyenadna,nt,dnabert_s
$RV src/regulatory/train_probes.py --models evo2 --out_dir data/regulatory/results_evo2
# merge evo2 rows into data/regulatory/results/probe_results.tsv (see prior session)
$RV src/regulatory/baselines.py
$RV src/regulatory/bootstrap_ci.py
$RV src/regulatory/paired_bootstrap.py
$RV src/regulatory/umap_visualization.py --model all

# 5. Update docs/results/regulatory_results.md and regulatory_paper.md with the
#    new (tighter) numbers; re-check the paired-bootstrap p-values.
```

## Gotchas / notes
- **Re-extraction is mandatory** after re-building windows — embeddings are row-
  aligned to `windows.tsv`; a new window set invalidates old embeddings.
- Evo 2 dominates the time/cost (A100 ~$2.50/h). NT/HyenaDNA/DNABERT-S are fast on
  A10G with the block-buffered writer.
- Keeping the old n=290 results (separate dir) lets you show CIs tightening with n.
- If GC-matching still caps below target, raise `--pool_size` further or relax
  `--gc_bins`; confirm classes stay balanced and GC-matched (audit `gc_content`).
- Evo 2 UMAP currently uses its *last* block (its weakest); regenerate at block 19
  if used in a figure.

## Other (smaller) strengthening ideas, if wanted
- Per-cCRE-class breakdown (PLS/pELS/dELS/CTCF) — requires carrying `cCRE_class`
  through `build_windows.py` (small change) → richer UMAP / per-class probes.
- Caduceus: retry with a prebuilt CUDA Docker base pinned to a known-good
  mamba-ssm/torch combo (current blocker in paper appendix).
