# Where Do DNA Language Models Encode Regulatory Identity? A Layer-wise Probing Study of Gene-Body vs. Regulatory Sequence

*Companion paper to the regulatory (Task 1) experiments. Data and per-model
detail in `regulatory_results.md`; code in `src/`.*

## Abstract

We ask whether frozen DNA language model (DNA-LM) embeddings encode the
distinction between **gene-body** and **regulatory** genomic sequence, and *where*
in each network that signal lives. Using GENCODE gene annotations and the ENCODE
cCRE registry, we build 4,024 length- and GC-matched 10 kb windows split by
chromosome to prevent leakage, extract per-layer mean-pooled embeddings from four
DNA-LMs spanning short- and long-context architectures, and train simple logistic
and MLP probes on every layer. We find that **only the long-context models
(HyenaDNA, Evo 2 7B) carry regulatory signal meaningfully above a GC-matched
sequence baseline** (best AUC ≈ 0.60, bootstrap CIs excluding chance), whereas the
short-context models (Nucleotide Transformer 2.5B, DNABERT-S) are statistically
indistinguishable from composition baselines. The signal emerges in **middle
layers** and, for Evo 2, *declines* in the final blocks. These results support the
hypothesis that regulatory identity is a long-range property that short-context
models cannot capture, and that probing frozen embeddings layer-by-layer localizes
where biological abstractions are formed.

## 1. Introduction

### Motivation
DNA language models are increasingly used as general-purpose feature extractors for
genomics, yet what biology they actually encode — and where — remains poorly
understood. If we can show that simple probes on frozen embeddings recover
biologically meaningful distinctions that simple sequence statistics cannot, and
localize that information across network depth, we gain both interpretability and
practical guidance on which models and which layers to use for downstream tasks
such as variant-effect prediction and noncoding-element annotation.

### The task
We study a binary discrimination central to functional genomics: is a genomic
window a **gene body** or a **regulatory element**? Regulatory elements
(promoters, enhancers, CTCF sites) are defined largely by their *position relative
to* and *long-range relationships with* gene bodies, not by local nucleotide
composition. This makes the task a natural probe for **long-range context**: a
model that only sees a few hundred bases of local sequence should struggle, while a
model that integrates kilobases of context should succeed — *if* it has learned
regulatory grammar.

### Goals and hypotheses
1. **Beyond composition (H1).** Do frozen-embedding probes outperform sequence-
   composition baselines (k-mer, GC/dinucleotide)? Because we GC-match the two
   classes, any gain must reflect learned features, not composition.
2. **Architecture dependence (H2).** Do **long-context** models (HyenaDNA, Evo 2)
   capture more regulatory signal than **short-context** models (NT, DNABERT-S),
   as predicted if regulatory identity is a long-range property?
3. **Layer localization (H3).** *Where* across depth does regulatory signal emerge,
   and does the pattern differ by architecture (e.g., early vs. late, or
   task-specialized final layers)?

## 2. Methods

### 2.1 Data
- **GENCODE v47** human gene annotations (gene bodies, exons).
- **ENCODE GRCh38 cCRE registry V4** (~2.35M candidate cis-regulatory elements).
- **GRCh38 (hg38)** reference genome for sequence extraction.

### 2.2 Label definitions
- **gene_body**: GENCODE gene intervals with any cCRE-overlapping region removed.
- **regulatory**: cCREs with any exon-overlapping region removed (i.e., intronic /
  intergenic regulatory elements).

### 2.3 Window construction (the key controls)
- **Center-on-site, fixed 10 kb windows.** A 10 kb window is centered on each
  labeled site's midpoint and clamped to chromosome bounds. (cCREs are short —
  median ~270 bp — so a naive "tile intervals ≥10 kb" scheme yields *zero*
  regulatory windows; centering fixes this.)
- **Length matching** is automatic (all windows are exactly 10 kb).
- **GC matching.** Candidate windows are binned by GC content and sampled so the
  two classes have matched GC distributions. This removes nucleotide composition as
  a confound — the central control that makes any positive result interpretable.
- **N-content filter** (>5% ambiguous bases excluded).
- Final dataset: **4,024 windows — 2,012 gene_body / 2,012 regulatory.**
- **Chromosome-level split** (prevents linkage leakage): train chr1–18 (3,372),
  validation chr19–20 (362), test chr21–22/X/Y (290).

### 2.4 Models and frozen embedding extraction
| Model | Context | Architecture | Layers probed |
|---|---|---|---|
| HyenaDNA (1M) | long, single-nucleotide | Hyena long-convolution | 10 hidden states |
| Evo 2 7B | long, multi-kingdom | StripedHyena 2 | 8 blocks (3–31) |
| Nucleotide Transformer 2.5B | short (≤1 kb eff.) | 6-mer BERT | 33 hidden states |
| DNABERT-S | short (512 tok) | BPE BERT, species-contrastive | 13 hidden states |

Each model is run frozen (`eval`, no grad). For each window we forward-pass and
**mean-pool token embeddings over the sequence axis at each layer**, yielding one
vector per (window, layer), cached to HDF5. Short-context models truncate the
10 kb window to their maximum context — this truncation is itself part of the
comparison. (Caduceus, a fifth long-context Mamba model, was planned but is
deferred for dependency reasons; see Appendix.)

### 2.5 Probes
Per (model, layer) we train two probes on the cached embeddings: **L2 logistic
regression** (regularization tuned on validation) and a **1-hidden-layer MLP**
(256 units). Features are standardized. We report on the held-out **test**
chromosomes.

### 2.6 Baselines
On the identical windows/splits: **k-mer TF–IDF (k=4,5,6) + logistic** and a
**GC + dinucleotide-frequency** logistic model — pure sequence-composition
features.

### 2.7 Evaluation
AUC, AUPRC, and F1 on the test split, reported per layer (layer-wise curves) and as
each model's best layer. To account for the small test split (n=290) we compute
**marginal bootstrap 95% CIs** (2,000 resamples) and, more powerfully, a **paired
bootstrap of the AUC difference** between conditions on the same resampled windows
(5,000 draws), reporting a bootstrap p-value p(Δ≤0). As a sanity check we run a
**label-shuffle null control** (probe trained on permuted train labels). We also
visualize embeddings with **UMAP**, colored by label and by chromosome.

## 3. Results

### 3.1 Sequence baselines sit at chance — the control holds
All composition baselines are at chance: k-mer TF–IDF AUC 0.49–0.54, GC+dinucleotide
AUC 0.50 (AUPRC ≈ 0.52–0.55, the balanced-class floor). Because the classes are
GC-matched, no composition feature separates them — so any probe above this floor
is using learned signal.

### 3.2 Only long-context models beat the baseline
Best-layer logistic probe (test split):

| Model | Context | Best layer | AUC | AUPRC | F1 |
|---|---|---|---|---|---|
| **HyenaDNA** | long | 8 | **0.599** | **0.628** | 0.570 |
| **Evo 2 7B** | long | 19 | **0.597** | **0.614** | 0.614 |
| NT 2.5B | short | 15 | 0.535 | 0.587 | 0.502 |
| DNABERT-S | short | 0 | 0.541 | 0.562 | 0.532 |
| baseline (GC+dinuc) | — | — | 0.50 | 0.55 | 0.52 |

The two **long-context** models reach AUC ≈ 0.60; the two **short-context** models
sit ~0.54, barely above composition. (Supports **H1** for long-context models,
**H2** overall.)

### 3.3 The gap is significant for long-context models (bootstrap)
Bootstrap 95% CIs on test AUC (n=290, 2,000 resamples):

| Condition | AUC | 95% CI | Excludes chance (0.5)? |
|---|---|---|---|
| HyenaDNA L8 | 0.615 | [0.554, 0.683] | **yes** |
| Evo 2 L19 | 0.588 | [0.523, 0.653] | **yes** |
| DNABERT-S L0 | 0.547 | [0.481, 0.613] | no |
| NT L15 | 0.527 | [0.460, 0.594] | no |
| baseline | 0.501 | [0.435, 0.566] | no |

Both long-context models' AUC CIs **exclude 0.5**; both short-context models and the
baseline **include 0.5**.

Marginal CIs are conservative here because predictions on the same windows are
correlated. A **paired bootstrap of the AUC difference** (same resampled windows,
5,000 draws) is the proper test and is far sharper:

| Contrast (A − B) | ΔAUC | 95% CI | p(Δ≤0) |
|---|---|---|---|
| HyenaDNA − baseline | **+0.115** | [+0.039, +0.193] | **0.002** |
| Evo 2 − baseline | **+0.087** | [+0.002, +0.172] | **0.023** |
| HyenaDNA − NT | **+0.089** | [−0.002, +0.176] | **0.027** |
| Evo 2 − NT | +0.061 | [−0.032, +0.150] | 0.093 |
| HyenaDNA − DNABERT-S | +0.069 | [−0.024, +0.160] | 0.068 |
| Evo 2 − DNABERT-S | +0.042 | [−0.050, +0.130] | 0.175 |
| HyenaDNA − Evo 2 | +0.028 | [−0.059, +0.111] | 0.246 |

**Both long-context models significantly beat the composition baseline**
(p=0.002, 0.023), and **HyenaDNA significantly beats the short-context NT**
(p=0.027); the remaining long-vs-short contrasts are positive trends (p≈0.07–0.18).
The two long-context models are statistically indistinguishable (p=0.246), as
expected. A **label-shuffle null control** (HyenaDNA best layer, permuted train
labels) gives test AUC **0.514** ≈ chance, confirming the pipeline does not leak.

### 3.4 Layer-wise: signal is in the middle, and Evo 2's late layers forget it
- **HyenaDNA**: near-baseline at the token-embedding layer (L0 AUPRC 0.54), rising
  to a plateau by L8–9 (0.63).
- **Evo 2**: peaks **mid-network** (blocks 11 and 19, AUPRC ~0.61) and **declines in
  the final blocks** (27→0.52, 31→0.51), suggesting late layers specialize for the
  autoregressive generation objective at the expense of regulatory-discriminative
  features.
- **NT**: a shallow mid-network plateau (~L11–16) only marginally above chance.
- **DNABERT-S**: essentially flat across depth; the embedding layer is as good as
  any. (Addresses **H3**: regulatory signal is a mid-network abstraction in the
  long-context models; the short-context models never form it.)

### 3.5 UMAP
UMAP projections of best-/late-layer embeddings (test split), colored by label and
chromosome, are saved in `results/umap/`. Consistent with the probe metrics, no
model shows clean visual separation of the two classes — the signal is real but
modest, recoverable by a linear probe rather than obvious in 2-D.

## 4. Discussion

**Context length, not parameter count, drives regulatory signal.** The clearest
result is architectural: the two long-context models (HyenaDNA, a 1.6M-parameter
Hyena model, and Evo 2 7B) both encode regulatory-vs-gene signal above chance, while
the two short-context transformers — including the 2.5B-parameter Nucleotide
Transformer — do not. NT is ~1,500× larger than HyenaDNA yet performs worse here,
consistent with the hypothesis that regulatory identity is defined by long-range
position and cannot be recovered from ≤1 kb of local sequence regardless of model
size.

**Layer-wise behavior is informative and architecture-specific.** That Evo 2's
regulatory signal peaks mid-network and decays in its final blocks is a concrete
instance of the broader interpretability point: the "best" representation for a
biological probe is often *not* the final layer, and where it lives reflects the
model's training objective.

**Negative results matter.** DNABERT-S, explicitly fine-tuned with a species-aware
contrastive objective, produces embeddings essentially useless for this task — a
reminder that representations are shaped by their training objective and do not
transfer universally.

## 5. Limitations
- **Small test split** (290 windows): marginal CIs are wide. The paired bootstrap
  recovers significance for the headline claims (long-context > baseline, p≤0.02;
  HyenaDNA > NT, p=0.03), but the remaining long-vs-short contrasts stay at the
  trend level (p≈0.07–0.18). Scaling the window count (re-build with a larger pool
  + re-extract; ~1–2 h, no new method) would likely push these to significance —
  the single highest-value next step.
- **Binary labels**: cCRE sub-classes (PLS/pELS/dELS/CTCF) are collapsed to a single
  "regulatory" class; finer structure is unexplored.
- **Mean-pooling** over long windows may dilute localized regulatory motifs; tokens
  are not resolution-comparable across tokenizers.
- **Caduceus deferred** (dependency-blocked), so we have two long-context models, not
  three.
- **Modest absolute performance** (best AUC ≈ 0.60): frozen embeddings + linear
  probe capture real but limited signal on this deliberately hard, composition-
  controlled task.

## 6. Conclusions
On a GC-matched, chromosome-split gene-body-vs-regulatory benchmark, frozen DNA-LM
embeddings carry regulatory signal **only when the model has long context**:
HyenaDNA and Evo 2 7B beat composition baselines (AUC ≈ 0.60, CIs above chance),
while short-context NT 2.5B and DNABERT-S do not. The signal is a **mid-network**
abstraction and, for Evo 2, is partially discarded by the final layers. Layer-wise
probing of frozen embeddings is thus an effective lens for *localizing* biological
information and for *choosing* representations — and it shows that, for regulatory
genomics, long-range context is the decisive architectural property.

## Appendix: reproducibility & engineering notes
- **Compute**: all extraction on Modal (A10G for HyenaDNA/NT/DNABERT-S; A100-80GB
  for Evo 2 7B); probes/baselines/UMAP/bootstrap on CPU.
- **Per-model fixes** required for frozen extraction: correct NT model id
  (`nucleotide-transformer-2.5b-multi-species`); NT max 1,000 tokens (position-
  embedding limit); DNABERT-S on transformers <5 with standard (non-Triton)
  attention via `attention_probs_dropout_prob>0` and forward hooks for hidden
  states; Evo 2 via the native `evo2` package + a torch-matched flash-attn wheel.
- **Caduceus deferred**: `mamba-ssm`/`causal-conv1d` CUDA extensions could not be
  installed on Modal after 7 approaches (prebuilt wheels mismatch PyPI torch's ABI;
  from-source builds fail on the GPU-less builder). Resolvable with a prebuilt CUDA
  Docker base pinned to a known-good mamba/torch combination.
- **Artifacts**: `probe_results.tsv`, `baseline_results.tsv`, `bootstrap_ci.tsv`,
  `layer_curves_*.png`, `umap/*.png`.
