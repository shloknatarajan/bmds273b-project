# Regulatory Task (Task 1) — Findings & Restart Handoff

**Date:** 2026-05-30
**Purpose:** Checkpoint before a terminal restart (with skip-permissions). This
is a self-contained handoff: state, results, fixes, and exact resume commands.
Compute path = **Modal** (driven from this EC2 box via `.env` tokens).

---

## TL;DR — where we are

- ✅ **Data + windows built** (the original blockers are fixed): `windows.tsv` =
  **4,024 windows**, 2,012 gene_body / 2,012 regulatory, GC-matched,
  chromosome-split (train 3,372 / val 362 / test 290).
- ✅ **HyenaDNA**: full embeddings (10 layers) + probes done. **Beats baseline.**
- ✅ **NT 2.5B**: embeddings extracting on Modal — **~1,460/4,024 when this was
  written; the run will be KILLED by the terminal restart** (see Resume below).
- ✅ **Baselines** run; sit at chance (the GC-match validation).
- ⏸ **DNABERT-S, Caduceus, Evo 2**: deferred "hard dependency" tier (old custom
  CUDA/Triton kernels vs the modern torch 2.7 stack). Root causes documented.

---

## RESULTS SO FAR

### HyenaDNA — regulatory vs gene-body probes (test split, 290 windows)
Layer-wise (logistic + MLP). Best logistic layer = **08**:

| | F1 | AUC | AUPRC |
|---|---|---|---|
| HyenaDNA layer 08 (logistic) | 0.570 | 0.599 | **0.628** |
| HyenaDNA layer 09 (logistic) | 0.553 | 0.601 | 0.627 |
| (signal rises with depth; layers 0–4 ≈ chance) | | | |

### Sequence baselines (no embeddings — chance-level by design)
| Baseline | F1 | AUC | AUPRC |
|---|---|---|---|
| 4-mer TF-IDF | 0.559 | 0.543 | 0.538 |
| 5-mer TF-IDF | 0.529 | 0.495 | 0.512 |
| 6-mer TF-IDF | 0.552 | 0.509 | 0.521 |
| GC + dinucleotide | 0.520 | 0.502 | 0.554 |

**Interpretation:** baselines ≈ chance (AUC ~0.50) confirms GC-matching removed
the composition shortcut, so HyenaDNA's AUPRC 0.628 / AUC 0.60 is **real signal
above sequence composition**, concentrated in deeper layers — exactly the
proposal's hypothesis. Modest but genuine; the 290-window test split makes it
noisy (bootstrap CIs still TODO).

Result files (local + should also be committed to Volume):
`data/regulatory/results/probe_results.tsv`, `baseline_results.tsv`,
`layer_curves_hyenadna.png`, `baseline_comparison.png`.

---

## BUGS FIXED THIS SESSION (all root-caused, tested where possible)

1. **`src/regulatory/build_windows.py`** — was producing ZERO regulatory windows
   and OOMing. Added `center_window()` (center a 10 kb window on each site, not
   require the site ≥10 kb), `build_candidates()` (coords-only, deduped, cheap),
   `materialize()` (fetch sequence only for a bounded `--pool_size` pool). Unit
   tests in **`tests/test_build_windows.py`** (10 passing) — run with
   `/home/ec2-user/regvenv/bin/python -m pytest tests/ -q`.
2. **`src/extract_embeddings.py`** — rewritten to one-sequence-per-pass (no
   batching → no attention-mask issues), load without `device_map` +
   `low_cpu_mem_usage=False`, **fixed NT model id** to
   `InstaDeepAI/nucleotide-transformer-2.5b-multi-species` (the `v2-2500m` id
   does not exist), NT `max_tok=1000` (position-embedding limit), and probe-based
   HDF5 sizing (HyenaDNA returns more hidden states than config says).
3. **`src/regulatory/baselines.py`** — `TfidfVectorizer(lowercase=False)`; the
   default lowercased k-mers so the uppercase `[ACGTN]+` token_pattern matched
   nothing (empty vocabulary).

---

## MODAL SETUP (`src/modal_app.py`)

- App `bmds273b-regulatory`; Volume `bmds273b-reg-data` at `/vol`
  (data + embeddings persist here).
- Functions: `download` (CPU), `build` (CPU 32 GB), `extract` (A10G, gpu_image:
  torch 2.7.1 + transformers 5.x — works for HyenaDNA + NT), `extract_dnabert`
  (A10G, transformers `<5` image).
- `main` local entrypoint: `--stage extract|smoke`, `--models`, `--limit`;
  fans out models concurrently, resilient per-model try/except.
- **Caduceus is intentionally NOT in this file** (its broken image would block
  everything, since Modal builds all registered images at `modal run`).

### Environment notes for the fresh session
- `.venv/bin/modal` = Modal client. `/home/ec2-user/regvenv` = CPU env with
  pyranges/pyfaidx + now **scikit-learn/h5py/matplotlib/umap-learn/pytest** (for
  downstream + tests).
- Always `set -a; . ./.env 2>/dev/null; set +a` first (Modal + HF tokens).
- Embeddings live on the Volume; `windows.tsv` and HyenaDNA embeddings are also
  pulled locally under `data/regulatory/`. **Gotcha:** `modal volume get <dir>`
  nests into `<local>/<name>/<name>/` — flatten after pulling.

---

## RESUME AFTER RESTART (do these, in order)

```bash
cd /home/ec2-user/bmds273b-project
set -a; . ./.env 2>/dev/null; set +a

# 1. NT was interrupted by the restart — RESUME it from its .cursor (continues
#    from where it stopped, not from 0). --resume reopens the HDF5s in r+ mode.
.venv/bin/modal run src/modal_app.py --stage extract --models nt --resume
#    (omit --resume to re-extract from scratch.)

# 2. Pull NT embeddings locally and flatten the nested dir
mkdir -p data/regulatory/embeddings/nt
.venv/bin/modal volume get bmds273b-reg-data regulatory/embeddings/nt/ data/regulatory/embeddings/nt/ --force
#   then: mv data/regulatory/embeddings/nt/nt/* data/regulatory/embeddings/nt/  (if nested)

# 3. Probes + UMAP across HyenaDNA + NT
/home/ec2-user/regvenv/bin/python src/regulatory/train_probes.py --models hyenadna,nt
/home/ec2-user/regvenv/bin/python src/regulatory/umap_visualization.py --models hyenadna,nt
```

NT re-run is the only lost work from the restart (HyenaDNA, windows, baselines
are all saved).

---

## DEFERRED "HARD DEPENDENCY" TIER (tackle last, as a batch)

All three are old models with custom kernels that fight torch 2.7 / triton 3.x /
transformers 5.x. They each need an **isolated old-stack image**:

- **DNABERT-S**: transformers `<5` fixed the meta-device crash, but its bundled
  `flash_attn_triton.py` calls `tl.dot(..., trans_b=True)` — removed in Triton
  3.x. Needs Triton ~2.0 ⇒ torch ~2.0/2.1 image, OR patch to disable its Triton
  flash-attention path (eager attention).
- **Caduceus**: `mamba-ssm`/`causal-conv1d` — PyPI wheel had an ABI mismatch
  (`undefined symbol`); source build fails on Modal's GPU-less builder. Fix:
  install **torch-matched prebuilt wheels** from the official GitHub releases
  (pick a torch version with cu/cp312 wheels), in its own `modal_caduceus.py`.
- **Evo 2**: A100-class GPU + port the native `evo2` loader from
  `src/species/04_extract_embeddings_evo2.py` (HF `AutoModelForCausalLM` path is
  broken). Plan Phase 7.

---

## REMAINING TODO (priority order)
1. Re-run NT (restart killed it) → probes + UMAP on HyenaDNA + NT.
2. Bootstrap CIs over windows (test split is only 290 — metrics are noisy).
3. Hard tier: DNABERT-S (old-stack image) → Caduceus (prebuilt wheels) → Evo 2.
4. Optional: 2–3× the window count (`build --pool_size 150000 --n_windows 8000`)
   for tighter metrics; re-extract.
5. Optional: batch sequences in `extract_embeddings.py` to cut NT runtime.
