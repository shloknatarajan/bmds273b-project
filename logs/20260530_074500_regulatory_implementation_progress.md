# Regulatory Task (Task 1) — Implementation Progress

**Date:** 2026-05-30 (session continuing from the Modal execution plan,
`logs/20260530_070337_regulatory_task_plan.md`)
**Scope of this log:** code written + Modal infrastructure stood up. Compute
path = **Modal** (decided this session), 4 non-Evo2 models run **concurrently**.

---

## 1. Fixed the window-building bug (`src/regulatory/build_windows.py`)

The original pipeline produced **zero regulatory windows** and OOM-killed on the
7.6 GB box. Both problems are now fixed, test-driven (TDD).

### New / changed functions
- **`center_window(start, end, window_size, chrom_len)`** — centers a 10 kb
  window on each site's midpoint, clamped to chromosome bounds. Replaces the old
  "skip any interval shorter than `window_size`" logic that dropped every cCRE
  (all ≤350 bp). Returns `None` if the chromosome is shorter than the window.
- **`build_candidates(df, label, chrom_lens, window_size)`** — builds
  **coords-only** candidate windows (no sequence yet), one per labeled site,
  deduped by `(chrom, win_start)`. Stays cheap even for ~2.3M cCREs. Skips
  unknown / too-short chromosomes.
- **`materialize(candidates, genome, max_n_frac)`** — fetches sequence + GC and
  applies the N-content filter **only for the bounded, subsampled pool**, so
  memory stays flat regardless of total site count. Replaced the OOM-prone
  `tile_intervals` (removed).
- **`main()` rewired:** load annotations → `build_candidates` per class →
  shuffle + cap to `--pool_size` (new arg, default 40k/class) → `materialize` →
  existing `gc_matched_sample` → write `windows.tsv` + `split_stats.txt`.
  `chrom_lens` now derived from the FASTA (`len(genome[name])`).

### Tests — `tests/test_build_windows.py` (10 passing)
Run with: `/home/ec2-user/regvenv/bin/python -m pytest tests/ -q`
(pytest installed into `regvenv`, which already has pyranges/pyfaidx).
- `center_window`: small-site → full window, centering, clamp at start, clamp at
  end, None when chrom < window.
- `build_candidates`: one window per site, dedup of windows clamped to same
  start, skips unknown chrom, skips too-short chrom.
- end-to-end: `build_candidates → materialize → gc_matched_sample` against a
  synthetic 60 kb FASTA (verifies sequence fetch, full-length seqs, GC values,
  equal per-class counts after matching).

---

## 2. Modal app (`src/modal_app.py`)

EC2 box = control plane; heavy RAM + all GPU work runs in Modal containers,
driven from here via `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in `.env`.

### Components
- **App:** `bmds273b-regulatory`.
- **Volume:** `bmds273b-reg-data` mounted at `/vol`; data under
  `/vol/regulatory`, embeddings under `/vol/regulatory/embeddings/<model>`.
- **Images:**
  - `cpu_image` — debian-slim + pandas/numpy/pyranges 0.1.4/pyfaidx/biopython
    (download + build).
  - `gpu_image` — debian-slim + torch 2.7.1 + transformers + accelerate/einops/
    h5py (HyenaDNA, NT, DNABERT-S).
  - `caduceus_image` — **isolated** on `nvidia/cuda:12.4.1-devel` base with
    `mamba-ssm` + `causal-conv1d` (`--no-build-isolation`). Separate so its
    fragile CUDA build can't break the other 3 models.
- Repo `src/` mounted into every image (`add_local_dir`) so the functions invoke
  the existing CLIs (`build_windows.py`, `extract_embeddings.py`) via subprocess.

### Functions / entrypoints
- `download` (CPU) — runs `download_data.sh` into the Volume, pyfaidx-indexes
  hg38, `vol.commit()`.
- `build` (CPU, **memory=32 GB**) — runs `build_windows.py` against the Volume →
  `windows.tsv`. The 32 GB allocation is what sidesteps the original OOM.
- `extract` (GPU, A10G) — runs `extract_embeddings.py` for one model; supports a
  `limit` arg for a cheap smoke slice.
- `main` local entrypoint — `--stage extract|smoke`, `--models`, `--limit`.
  **Fans out all 4 models concurrently** (spawn-all-then-gather): each lands in
  its own A10G container, so wall-clock ≈ slowest single model.
- `MODEL_CFG` sets per-model image / GPU / batch size (NT batch 4, DNABERT-S 8,
  Caduceus 2, HyenaDNA 1).

### Validation done (no spend)
- `pytest tests/` → 10 passed.
- Module imports cleanly; `app.registered_functions` = `build, download, extract`.
- `modal run src/modal_app.py --help` parses (entrypoint args shown);
  env-token auth works.

---

## 3. Modal execution status

- **Images:** `cpu_image` and `gpu_image` built successfully on Modal
  (transformers resolved to **5.9.0**, torch 2.7.1 — watch for model-loading
  incompatibility under transformers 5.x during extract). `caduceus_image`
  build deferred until the Caduceus extract call.
- **`download`:** launched on Modal (CPU). Pulls GENCODE v47 GTF, ENCODE GRCh38
  cCREs, hg38.fa (+ decompress + pyfaidx index) into the Volume. Status at time
  of writing: in progress / not yet confirmed complete.

### Usage (from repo root, after `set -a; . ./.env; set +a`)
```bash
.venv/bin/modal run src/modal_app.py::download              # data → Volume
.venv/bin/modal run src/modal_app.py::build                 # windows.tsv (32 GB)
.venv/bin/modal run src/modal_app.py --stage smoke --models dnabert_s
.venv/bin/modal run src/modal_app.py --stage extract        # 4 models concurrent
```

---

## 4. Next steps
1. Confirm `download` finished; then run `build` and **verify `windows.tsv` has
   both classes + matched GC** (`split_stats.txt`) before GPU spend.
2. Run the 4-model concurrent `extract`; watch the Caduceus `mamba-ssm` build
   and any transformers-5.x loader issues.
3. Pull embeddings back (`modal volume get`) and run probes / baselines / UMAP
   (plan Phases 4–6).
4. Evo 2 deferred to plan Phase 7 (A100 + native `evo2` loader port).

## 5. Open risks
- transformers **5.9.0** vs scripts written for 4.x (`trust_remote_code` models:
  HyenaDNA, Caduceus, DNABERT-S) — possible loader breakage; pin transformers in
  the image if so.
- Caduceus `mamba-ssm` / `causal-conv1d` CUDA build is the most fragile step.
- `windows.tsv` must be confirmed non-degenerate (the original bug produced 0
  regulatory) before spending on GPUs.
