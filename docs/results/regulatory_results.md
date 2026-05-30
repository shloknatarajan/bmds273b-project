# Regulatory vs. Gene-Body Classification — Results

**Task 1 of the proposal** (`docs/project_proposal.md`): probe frozen DNA-LM
embeddings to classify 10 kb genomic windows as **gene body** vs **regulatory**,
layer-by-layer, and ask whether learned embeddings beat sequence-composition
baselines and *where* regulatory signal emerges across model depth.

*Living document — updated as each model completes. Last update: 2026-05-30.*

---

## Proposal questions this addresses

1. **Do simple probes on frozen embeddings outperform sequence-based baselines?**
   (proposal "feasible target")
2. **Where does regulatory signal emerge across layers**, and are there
   **architecture-dependent patterns** — e.g. long-context (HyenaDNA, Caduceus,
   Evo 2) vs short-context (NT, DNABERT-S) models?

(The embedding-distance / phylogenetic question is Task 2 / species, done by
another team — not covered here.)

---

## Dataset & method

- **Windows:** 4,024 total — 2,012 gene_body / 2,012 regulatory, **GC-matched**
  (so composition is not a giveaway), each a 10 kb window centered on the site.
- **Labels:** `gene_body` = GENCODE gene minus cCRE overlap; `regulatory` =
  ENCODE cCRE minus exon overlap (intronic/intergenic). Binary.
- **Split (chromosome-level, anti-leakage):** train chr1–18 (3,372) / val
  chr19–20 (362) / test chr21–22,X,Y (290).
- **Embeddings:** each model frozen, `output_hidden_states`, mean-pool over the
  sequence per layer → one vector per (window, layer). One HDF5 per layer.
- **Probes:** per (model, layer) — L2 logistic regression (C swept on val) and a
  1-hidden-layer MLP; features standardized. Metrics on the **test** split.
- **Compute:** Modal (A10G GPUs); see `src/modal_app.py`.

---

## Baselines (sequence features, no embeddings) — the interpretability anchor

| Baseline | F1 | AUC | AUPRC |
|---|---|---|---|
| 4-mer TF-IDF | 0.559 | 0.543 | 0.538 |
| 5-mer TF-IDF | 0.529 | 0.495 | 0.512 |
| 6-mer TF-IDF | 0.552 | 0.509 | 0.521 |
| GC + dinucleotide | 0.520 | 0.502 | 0.554 |

**All baselines sit at chance (AUC ≈ 0.50).** This is the key control: because
windows are GC-matched, raw composition cannot separate the classes. Any probe
that beats AUC ≈ 0.50 / AUPRC ≈ 0.52 is using *learned* signal, not composition.

---

## Per-model results

### HyenaDNA (long-context, single-nucleotide; 10 hidden states) — ✅ done

Best **logistic** layer = **8**. Layer-wise (logistic):

| Layer | F1 | AUC | AUPRC |
|---|---|---|---|
| 0 | 0.570 | 0.531 | 0.543 |
| 1 | 0.587 | 0.582 | 0.628 |
| 2 | 0.578 | 0.577 | 0.612 |
| 5 | 0.544 | 0.554 | 0.603 |
| 8 | 0.570 | **0.599** | **0.628** |
| 9 | 0.553 | **0.601** | 0.627 |

**Findings:**
- **Beats baseline:** best AUC 0.60 vs ~0.50; best AUPRC 0.628 vs ~0.54.
  → embeddings carry real regulatory signal beyond composition (proposal Q1: yes,
  modestly, for HyenaDNA).
- **Layer trend:** layer 0 (token embedding) ≈ baseline (AUPRC 0.543); signal
  rises and peaks in **deeper layers (8–9)** — consistent with regulatory
  identity being a higher-level abstraction, not local composition.
- Effect is **modest** and the test split is small (290 windows) → metrics are
  noisy; bootstrap CIs still TODO.

### NT 2.5B (short-context, 6-mer transformer; 33 hidden states) — ✅ done

Best **logistic** layer = **15**: F1 0.502, AUC **0.535**, AUPRC **0.587**.
AUPRC rises slightly from ~0.54 (early) to a shallow mid-network plateau
(~0.57–0.59 around layers 11–16), then flat. Best MLP: layer 31, AUPRC 0.606.

**Findings:** only **marginally above baseline** (AUPRC 0.587 vs ~0.54; AUC 0.535
vs 0.50). A short-context (≤1 kb effective) model captures little regulatory-vs-gene
signal beyond composition — consistent with regulatory identity being a long-range
property.

### DNABERT-S (short-context, 512 tok; 13 hidden states) — ✅ done

Best **logistic** layer = **0**: F1 0.532, AUC 0.541, AUPRC 0.562. **Essentially
flat across depth** (AUPRC 0.51–0.56), no emergence trend; the embedding layer is
as good as any. Best MLP: layer 6, AUPRC 0.595.

**Findings:** the **weakest** model — barely above baseline (AUPRC 0.562 vs 0.554).
The species-contrastive pretraining objective gives DNABERT-S embeddings that don't
transfer to the regulatory task, and its short context can't see regulatory context.

### Evo 2 7B (long-context, multi-kingdom StripedHyena; blocks 3–31) — ✅ done

Best **logistic** block = **19**: F1 0.614, AUC **0.597**, AUPRC **0.614**.
AUPRC by block: 3=0.57, 7=0.56, **11=0.61**, 15=0.57, **19=0.61**, 23=0.57,
27=0.52, 31=0.51 — peaks **mid-network** then **drops in late blocks**.

**Findings:** clearly above baseline (AUC 0.597 vs 0.50), on par with HyenaDNA —
**the second long-context model to beat the short-context ones.** The late-block
decline suggests Evo 2's final layers specialize for its autoregressive generation
objective rather than retaining regulatory-discriminative features.

### Caduceus (long-context, bidirectional Mamba) — ⛔ deferred (dependency-blocked)
_`mamba-ssm` / `causal-conv1d` CUDA extension could not be installed on Modal after
7 approaches: prebuilt wheels (cu12 × torch 2.6/2.7 × cxx11abi TRUE/FALSE) all hit
`undefined symbol` (wheels mis-built vs PyPI torch ABI), and from-source builds
fail in Modal's GPU-less builder (`causal-conv1d` compile error, likely OOM across
GPU arches). Path forward: a prebuilt CUDA Docker base with a known-good
mamba/torch pin, or a builder with more RAM + single `TORCH_CUDA_ARCH_LIST`._

---

## Cross-model comparison (best logistic layer, test split)

| Model | Context | Best layer | AUC | AUPRC | vs baseline (AUPRC ~0.54) |
|---|---|---|---|---|---|
| **HyenaDNA** | long (single-nt) | 8 | **0.599** | **0.628** | **+0.09** ✅ |
| **Evo 2 7B** | long (multi-kingdom) | 19 | **0.597** | **0.614** | **+0.07** ✅ |
| NT 2.5B | short (6-mer) | 15 | 0.535 | 0.587 | +0.05 (marginal) |
| DNABERT-S | short (512 tok) | 0 | 0.541 | 0.562 | +0.02 (negligible) |
| Caduceus | long (Mamba) | — | — | — | ⛔ deferred (dep-blocked) |

### Bootstrap 95% CIs (test split, n=290; AUC, 2000 resamples)

| Condition | AUC median | 95% CI | Above chance (0.5)? |
|---|---|---|---|
| HyenaDNA L8 | 0.615 | [0.554, 0.683] | **yes** |
| Evo 2 L19 | 0.588 | [0.523, 0.653] | **yes** |
| DNABERT-S L0 | 0.547 | [0.481, 0.613] | no (CI spans 0.5) |
| NT L15 | 0.527 | [0.460, 0.594] | no |
| GC+dinuc baseline | 0.501 | [0.435, 0.566] | no (chance, as expected) |

Both long-context models' AUC CIs exclude 0.5; the short-context models' and the
baseline's do not. CIs are wide (small test split) so the long-vs-short gap is a
clear trend rather than a tight separation.

## Running summary vs proposal targets

| Question | Finding (so far) |
|---|---|
| Embeddings > sequence baselines? | **Yes, but architecture-dependent.** HyenaDNA clearly (AUC 0.60 vs 0.50); NT marginal; DNABERT-S negligible. |
| Layer-wise emergence | HyenaDNA peaks layers 8–9; NT a shallow mid-network plateau (~15); DNABERT-S flat (no emergence). |
| **Long vs short context** | **The headline result: the long-context HyenaDNA is the only model meaningfully above baseline; both short-context models barely beat composition.** Supports the proposal's hypothesis that regulatory identity is a long-range property. Evo 2 (long) will be a key second long-context datapoint. |

**Caveats:** effect sizes are modest (best AUC ≈ 0.60) and the test split is small
(290 windows) → metrics are noisy; bootstrap CIs are a planned strengthening step.

## Artifacts
- `data/regulatory/results/probe_results.tsv`, `baseline_results.tsv`
- `data/regulatory/results/layer_curves_hyenadna.png`, `baseline_comparison.png`
- Embeddings on Modal Volume `bmds273b-reg-data` under `regulatory/embeddings/<model>/`.
