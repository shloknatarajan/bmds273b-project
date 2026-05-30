# Regulatory Task (Task 1) — Execution Plan

**Date:** 2026-05-30T07:03:37+0000
**Author:** plan derived from the completed species pipeline (Task 2)
**Goal:** Take Task 1 (regulatory vs. gene-body classification) from
"scripts written, never run" to the same end-to-end, results-on-disk state
that Task 2 (species) already reached.

---

## 1. Reference: what the species pipeline (Task 2) actually proved out

Task 2 is the **working vertical slice**. It established a recipe we should
copy step-for-step. What exists and ran end-to-end (`src/species/`):

| Stage | Species script | Output produced |
|---|---|---|
| Download | `00_download_data.sh` | GTDB taxonomy/trees + NCBI genomes |
| Sample labels | `01_subsample_species.py` | `species_index.tsv` (200 species) |
| Build examples | `02_extract_fragments.py` | `fragments.tsv` (10k frags, split by **species**) |
| Ground-truth distance | `03_build_phylo_distance.py` | `phylo/distance_matrix.npz` |
| Embeddings (per model) | `04_extract_embeddings_{evo2,nt,dnabert_s}.py` | `embeddings/<model>/<layer>.h5` |
| Probes (per model) | `05_train_probes*.py` | `results/classification_results*.tsv` |
| UMAP (per model) | `06_umap_visualization*.py` | `results/umap*/*.png` |
| Distance eval | (in 05/utility) | `phylo_correlation*.tsv`, `phylo_curves.png` |

**Key proven design decisions to carry over verbatim:**
- One HDF5 per `(model, layer)`; **row order preserved** end-to-end so probes
  slice embeddings by split index with no join.
- Held-out split done on the **biological unit, not the example** (species for
  Task 2 → **chromosome** for Task 1). This is the anti-leakage rule.
- Mean-pool token embeddings per layer → one vector per example.
- Probe layer-wise (every layer), logistic + shallow MLP, standardize features,
  tune regularization on val.
- Run **Evo 2 via the native `evo2` package**, not `AutoModelForCausalLM`
  (see `04_extract_embeddings_evo2.py`). The HF path is broken.

---

## 2. Where Task 1 stands today (gap analysis)

Scripts exist and are reasonably designed (`src/regulatory/`), but **nothing
has been run** — there is no `data/regulatory/windows.tsv`, no embeddings, no
results. Two hard blockers were already diagnosed
(`logs/20260527_191556_regulatory_compute_decision.md`):

- ✅ **Data downloaded** (`data/regulatory/`): GENCODE v47 GTF, GRCh38 cCREs
  BED (2.35M cCREs), hg38.fa + `.fai`.
- ❌ **`build_windows.py` is buggy + OOMs.** Two independent problems:
  1. **Windowing logic produces ZERO regulatory windows.** `tile_intervals()`
     skips intervals shorter than `window_size` (10 kb); every cCRE is
     ≤350 bp, so no cCRE ever yields a window. Spec intent is to **center** a
     10 kb window *on* each site, not require the site to be ≥10 kb.
  2. **OOM on 7.6 GB RAM.** Loading 2.35M cCREs + 2.15M exons into pyranges
     and `.subtract()` exceeds memory → SIGKILL.
- ✅ **Compute decision made: Modal** (driven from this EC2 box via `.env`
  tokens). The 4 non-Evo2 models fit a single L40S/A10; Evo 2 needs an A100-80GB
  + the hard `evo2` install, kept in its own image.

So unlike Task 2, Task 1 is blocked at step 1 (data prep) and has never touched
the GPU stages.

---

## 3. The plan

Ordered by dependency. Phases 0–2 unblock everything; Phases 3–6 mirror the
species recipe.

### Phase 0 — Stand up the Modal app (blocking)
**Compute decision: Modal.** This box stays the control plane — Claude Code
drives Modal from here via the CLI using the `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`
already in `.env`. Heavy RAM and all GPU work run in Modal containers; the EC2
box only orchestrates and holds the lightweight CPU steps.

Set up:
1. **Modal app skeleton** (`src/modal_app.py` or `src/regulatory/modal_app.py`):
   - A base **image** = the `pyproject.toml` deps (torch cu128, transformers,
     pyranges/pyfaidx, scikit-learn, h5py, umap-learn) installed into the image.
   - A persistent **Volume** (`reg-data`) mounted at e.g. `/vol`, holding
     `data/regulatory/` (annotations + hg38) and `embeddings/` outputs so they
     survive between function calls and across the 4-model sweep.
   - **GPU functions** sized per model (see Phase 3): `L40S`/`A10` for the 4
     non-Evo2 models, `A100-80GB` reserved for Evo 2 (Phase 7).
   - A **CPU function** with high memory (`memory=32768`) for `build_windows`
     (sidesteps the 7.6 GB OOM entirely — no need to fit the local box).
2. **Get data into the Volume.** Either run `download_data.sh` inside a Modal
   function writing to the Volume, or `modal volume put` the existing local
   `data/regulatory/` (~4.4 GB) up. Downloading inside Modal is usually faster.
- **Exit criteria:** `modal run` executes a trivial function that lists the
  populated Volume; tokens authenticate from this box.

### Phase 1 — Fix `build_windows.py` + run it as a Modal CPU function (blocking)
This is the single highest-leverage fix; everything downstream depends on
`windows.tsv` existing and being correct. On Modal the OOM blocker disappears
(give the function 32 GB), so the **correctness** fix is the must-do; the
memory-bounded sampling is still good hygiene and keeps the run cheap:
1. **Center-on-site windows.** For each labeled site (cCRE or gene-body
   interval), center a `window_size` (10 kb) window on its midpoint and clamp
   to chromosome bounds. This makes small cCREs usable → non-zero regulatory
   class.
2. **Memory-bounded sampling.** Don't materialize sequences for all candidates.
   - Subsample a candidate **pool** per class (~40k each).
   - Compute GC on the pool storing **only floats + coords** (never the 10 kb
     string).
   - GC-match the two classes down to `n_windows`/class.
   - Extract sequences with `pyfaidx` **only for the final ~10k selected
     windows.**
3. Keep the existing label definitions and chromosome split (chr1–18 train /
   19–20 val / 21,22,X,Y test) — these match the spec and Task 2's leakage rule.
4. Emit `windows.tsv` (to the **Volume**) with the columns the README documents
   (`window_id, seq, label, split, gc_content, chrom, start, end`) and
   `split_stats.txt`.
- **Exit criteria:** `windows.tsv` on the Volume with **both** classes present,
  balanced per split, GC distributions matched (audit `gc_content` by label).
  Sanity-check counts before spending GPU time.

### Phase 2 — Smoke-test the shared embedding extractor on Modal (small)
- Wrap `src/extract_embeddings.py` as a Modal GPU function reading
  `windows.tsv` from the Volume. Run on a **tiny slice** (e.g. 50 windows) for
  one easy model (DNABERT-S or NT) to confirm the windows.tsv → HDF5 path works
  and row order is preserved, before committing to full GPU runs.
- This is also where you validate the Modal image actually loads each model
  (Caduceus needs `mamba-ssm` + `causal-conv1d` CUDA builds in the image).
- Confirm `umap-learn` is actually installed/pinned (flagged missing from
  `pyproject.toml` in the progress log) since Phase 6 needs it.
- **Exit criteria:** one `layer_XX.h5` written for the slice, shape
  `(n, hidden_dim)`, indices align to windows.tsv rows.

### Phase 3 — Extract embeddings on Modal GPUs, the 4 non-Evo2 models first
Mirror Task 2's per-model approach. Run the Modal GPU function over the full
`windows.tsv` on the Volume:
- **HyenaDNA, Caduceus, NT v2 2.5B, DNABERT-S** — all fit a single **L40S/A10**
  (NT 2.5B is the largest at ~5 GB). **Decision: run all 4 concurrently** as
  separate Modal containers (`.spawn()` / `modal run` fan-out) — each gets its
  own GPU, wall-clock ≈ slowest single model rather than the sum. Note context
  limits: long-context models (HyenaDNA, Caduceus) use
  the full 10 kb; short-context (NT 2 kb, DNABERT-S 512) truncate — that
  truncation *is* part of the comparison the proposal wants.
- Cache one HDF5 per `(model, layer)` **to the Volume**. Sweep enough layers
  for real layer-wise curves (Task 2 lesson: don't keep only 1–2 layers, or
  results won't match the on-disk embeddings).
- **Defer Evo 2** to Phase 7.
- Cost: at ~$1–2/hr these four are a few dollars total; full 5-model pass est.
  **~$15–40** against the $30/mo free credits.
- **Exit criteria:** `embeddings/<model>/layer_*.h5` on the Volume for all 4.
  Pull them down to the EC2 box (`modal volume get`) for the CPU phases, or run
  Phases 4–6 as CPU Modal functions too — the HDF5s are small.

### Phase 4 — Train probes (mirror `05_train_probes`)
- Per `(model, layer)`: L2 logistic regression + shallow MLP on cached
  embeddings. Standardize features; tune regularization on **val**; report on
  **test**.
- Metrics: **AUPRC, AUC, F1** (binary: gene_body vs regulatory).
- Write `results/probe_results.tsv` as `(model, layer, probe, f1, auc, auprc)`.
- **Exit criteria:** `probe_results.tsv` populated for all 4 models × layers.

### Phase 5 — Sequence baselines (mirror the missing Task 2 baselines too)
- **k-mer TF–IDF (k=4–6) + logistic** and a **shallow 1D CNN** on one-hot
  sequence, on the same windows. Run on CPU (fast).
- This is what makes the embedding numbers interpretable ("do embeddings beat
  trivial composition?"). Note: Task 2 still lacks these — the `baselines.py`
  work here is reusable in spirit for both tasks.
- Write `results/baseline_results.tsv`.
- **Exit criteria:** baseline scores on disk for comparison plots.

### Phase 6 — UMAP + analysis plots (mirror `06_umap_visualization`)
- UMAP of embeddings colored by **label** (gene_body/regulatory) and by
  **chromosome**, per model at a few layers.
- `layer_curves_<model>.png` (metric vs layer), `model_comparison.png`
  (best-layer AUPRC across models), `baseline_comparison.png`.
- Add **bootstrap CIs over windows** (also still missing for Task 2).
- **Exit criteria:** plots in `results/` + `results/umap/`.

### Phase 7 — Evo 2 (deferred; A100 Modal function + native loader)
- Port the native-`evo2`-package loader from `src/species/04_extract_embeddings_evo2.py`
  into `src/extract_embeddings.py` (the current HF `AutoModelForCausalLM` path
  is known-broken for Evo 2).
- Build a **separate Modal image** for Evo 2 with the `evo2`/Vortex (StripedHyena 2)
  install — this is finicky and CUDA-specific, so keep it isolated from the
  Phase-3 image so it can't break the other 4 models. Run on a Modal
  **A100-80GB** function ($2.50/hr).
- Then re-run Phases 4–6 including Evo 2 for the full 5-model comparison.
- **Exit criteria:** Evo 2 embeddings + probes folded into the results tables
  and plots.

---

## 4. Differences from the species task (don't blindly copy)

- **No phylogenetic-distance analogue.** Task 1 is pure binary classification —
  there is no continuous ground-truth "distance" target, so skip the
  `03_build_phylo_distance` / `phylo_correlation` machinery entirely.
- **Leakage unit is the chromosome, not the species.** Already handled by the
  chromosome split.
- **GC/length matching is the make-or-break step.** For species, the analogue
  was held-out species; here, if classes differ in GC the probe learns
  composition, not biology, and results are uninterpretable. Audit this before
  any GPU spend.
- **Context-length truncation matters more.** Regulatory identity is positional
  / long-range, so the long- vs short-context model contrast is the headline
  result, not a side note.

## 5. Carried-over risks / open questions
- Confirm `windows.tsv` has a non-degenerate regulatory class **before** GPU
  spend (the original bug produced zero).
- `umap-learn` not pinned in `pyproject.toml` despite being used.
- Evo 2 install (`evo2`/Vortex) is the project's hardest dependency — isolate
  it to a dedicated Phase-7 Modal image so it never blocks the other 4 models.
- **Compute decision: Modal** (driven from this EC2 box via the `.env` tokens).
  Watch the $30/mo free-credit ceiling; the A100 Evo 2 run is the costly part.
- **Modal Volume is the source of truth** for `windows.tsv` + embeddings — keep
  row order identical to what probes expect, and don't let a local stale copy
  drift from the Volume.
- Build a Modal image once that satisfies the awkward CUDA builds (`mamba-ssm`,
  `causal-conv1d` for Caduceus); validate in Phase 2 before the full sweep.

## 6. Definition of done (Task 1)
- [ ] `data/regulatory/windows.tsv` + `split_stats.txt`, both classes, GC-matched.
- [ ] Embeddings for HyenaDNA, Caduceus, NT v2, DNABERT-S (and later Evo 2).
- [ ] `results/probe_results.tsv` (AUPRC/AUC/F1 per model × layer).
- [ ] `results/baseline_results.tsv` (k-mer TF–IDF + 1D CNN).
- [ ] Layer-wise curves, model comparison, baseline comparison, UMAP plots.
- [ ] Bootstrap CIs.
- [ ] Short writeup: where regulatory signal emerges by layer, long- vs
      short-context, embeddings vs. baselines.
