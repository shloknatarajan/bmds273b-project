# Regulatory Pipeline — Blocker Findings & Compute Decision

**Date:** 2026-05-27T19:15:56+0000
**Context:** Running the regulatory (Task 1) data pipeline. Merge resolved
(descriptive-named scripts kept). Data downloaded OK. Stuck at `build_windows.py`.

---

## What works so far

- ✅ Merge resolved; `src/regulatory/` has one coherent pipeline.
- ✅ CPU venv at `/home/ec2-user/regvenv` (pyranges 0.1.4, pyfaidx, biopython,
  pandas, numpy). gcc installed via `sudo yum install gcc` (needed to build
  `sorted-nearest`).
- ✅ Data downloaded to `data/regulatory/` (~4.4 GB):
  - `gencode.v47.annotation.gtf.gz` (59 MB) — 78,724 genes, 2,155,005 exons, 4.1M total lines
  - `GRCh38-cCREs.bed` (129 MB) — **2,348,854 cCREs** (SCREEN Registry V4)
  - `hg38.fa` (3.27 GB) + `.fai` index built

## Blocker 1 — build_windows.py OOM-killed (EXIT 137 / SIGKILL)

- **This box has only 7.6 GB RAM** (`free -h`: 7.6Gi total, ~5.8Gi available).
- Loading 2.35M cCREs + 2.15M exons into pyranges and running `.subtract()`
  exceeds available memory → OOM kill. Even a bare load+subtract test was killed.

## Blocker 2 — windowing logic produces ZERO regulatory windows (code bug)

- cCRE length distribution: **min 150, median 273, p99 350, max 350 bp.**
  Every cCRE is < 400 bp.
- `tile_intervals()` skips any interval shorter than `window_size` (default
  10,000 bp): `if (end - start) < window_size: continue`.
- Therefore **no cCRE ever produces a window → 0 regulatory windows.** The
  gene side (many genes > 10 kb) is what generated a huge candidate list and
  drove the memory blowup.
- **Spec intent** (`technical_steps.md` §2.3): "Sample matched ≥10 kb windows
  *around* each labeled site." Windows should be **centered on** each site,
  not require the site to be ≥10 kb.

### Required code fix (independent of compute choice)
Rewrite windowing in `build_windows.py` to:
1. Center a `window_size` window on each cCRE / gene-body site (clamp to
   chromosome bounds); this makes small cCREs usable.
2. Be memory-bounded: subsample a candidate **pool** (e.g. ~40k/class), compute
   GC on the pool storing only floats+coords (NOT the 10 kb sequence), GC-match
   down to `n_windows`/class, and extract sequences ONLY for the final ~10k
   selected windows. (Storing seq for all candidates = OOM even at 32 GB.)

---

## Compute options (DECISION PENDING)

The pipeline has two demanding stages:
- **build_windows**: needs ~16–32 GB RAM, no GPU.
- **embeddings** (`src/extract_embeddings.py`): needs a GPU. Must run somewhere
  with CUDA regardless.

| Option | What it means | Pros | Cons |
|---|---|---|---|
| **A. Resize this instance** to a bigger GPU type (e.g. g5.2xlarge: A10G 24 GB + 32 GB RAM) | I keep driving the same box directly; run existing scripts as-is | Simplest to operate; no Modal app; no evo2-install pain; one box for both stages | Brief downtime to resize; A10G 24 GB may be too small for Evo 2 7B (fine for the other 4 models); fast data re-download needed |
| **B. Modal** | I build a Modal app (image + Volume + GPU functions), driven from here with the `.env` tokens (`MODAL_TOKEN_ID`/`SECRET`) | Elastic A100s (good for Evo 2); per-second billing; no instance mgmt | More setup; Evo 2's `evo2` package can be finicky to install in a Modal image; data re-download into a Volume |
| **C. Split** | Lean-rewrite build_windows to fit 7.6 GB here; Modal only for GPU embeddings | No resize | Two environments; still need the windowing fix; more moving parts |

### Operability note
I run Claude Code **on this EC2 box**. So:
- "Resize this instance" = cleanest if it's the *same* box (I keep direct access).
- A brand-new separate EC2 would need Claude Code installed there or SSH access
  for me to drive it.
- Modal I can drive from here via the CLI + the tokens already in `.env`.

## Per-model requirements — Evo 2 vs the other four

The embedding step (`src/extract_embeddings.py`) treats 5 models uniformly, but
their compute/install needs are NOT uniform. **Evo 2 is the outlier that drives
any A100 requirement; the other 4 fit on one modest GPU.**

| Model | HF id (loader) | ~Params / weights | Max ctx (script) | GPU mem needed | Install | Fits A10G 24GB? |
|---|---|---|---|---|---|---|
| **Evo 2 7B** | `arcinstitute/evo-2-7b` (causal, fp16) | **7B / ~14 GB** | 8,192 tok | **A100 40–80GB / H100** | **Hard** — StripedHyena 2 via `evo2`/Vortex pkg; CUDA-specific build | **No** (tight at best) |
| NT v2 2.5B | `InstaDeepAI/...-2500m-multi-species` (masked, fp16) | 2.5B / ~5 GB | 2,048 tok | ~8–12 GB | Easy (standard HF) | **Yes** |
| Caduceus | `kuleshov-group/caduceus-ps_seqlen-131k...` (masked, fp32) | small (d=256, 16L) | 131,072 | <4 GB | **Medium** — needs `mamba-ssm` + `causal-conv1d` (CUDA build) | **Yes** |
| HyenaDNA | `LongSafari/hyenadna-large-1m-seqlen-hf` (base, fp32) | small (~6.6M) | 1,000,000 | <2 GB | Easy-ish (trust_remote_code) | **Yes** |
| DNABERT-S | `zhihan1996/DNABERT-S` (base, fp32) | ~117M | 512 tok | <2 GB | Easy (standard HF) | **Yes** |

### What this means
- **The 4 non-Evo2 models** (NT, Caduceus, HyenaDNA, DNABERT-S) all fit
  **comfortably on a single A10G 24 GB / L40S**, even simultaneously if run
  sequentially. NT 2.5B is the largest at ~5 GB. → **Option A (resize this box
  to g5.2xlarge) covers all four with no Modal needed.**
- **Evo 2 7B is the only model that:**
  1. needs an **A100-class GPU** (~14 GB weights + long-ctx activations — won't
     comfortably fit A10G 24 GB),
  2. has a **hard install** (`evo2`/Vortex, not plain transformers), AND
  3. is **likely broken as currently coded** — `extract_embeddings.py` loads it
     via `AutoModelForCausalLM`, but git history already notes "issue in evo2
     need to rewrite to not use hugging face model." The working species
     pipeline (`src/species/04_extract_embeddings_evo2.py`) uses the native
     `evo2` package instead.

### Recommended split
- **Phase 1 (now):** 4 HF models on a modest GPU (resized g5.2xlarge **or**
  Modal L40S/A10). Cheap, unblocks the whole probe/baseline/UMAP analysis.
- **Phase 2 (later):** Evo 2 on an **A100 (Modal A100 80GB, or a p4d/A100 EC2)**,
  AFTER porting the native-`evo2`-package loader from the species script into
  `extract_embeddings.py`. Do not block Phase 1 on this.

## Cost reference (Modal, per-hour; from modal.com/pricing 2026-05-27)
H100 $3.95 · A100 80GB $2.50 · A100 40GB $2.10 · L40S $1.95 · A10 $1.10 · L4 $0.80 · T4 $0.59.
$30/mo free credits (Starter). Full 5-model × both-tasks pass est. **~$15–40**.

## Recommendation
**Option A (resize this instance to a GPU box, ~32 GB RAM)** for simplicity,
running the 4 HF models first and deferring Evo 2. Use Modal (Option B/C) if you
specifically want A100-class GPUs for Evo 2 without managing an instance.

## Immediate next action (any path)
Fix `build_windows.py` windowing (center-on-site + memory-bounded sampling),
then run on a box with ≥16 GB RAM to produce `data/regulatory/windows.tsv`.
