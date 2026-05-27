# Modal Cost Estimate

Cost estimate for running the experiments in `technical_steps.md` on
[Modal](https://modal.com). Grounded in the actual workload defined in the
repo: 10,000 species fragments × 1,024 bp; 5,000 windows/class = 10,000
regulatory windows × 10 kb; 5 models.

## What drives the cost

- **GPU embedding extraction (step 4) is the only meaningful cost.** Probes
  (5), baselines (6), evaluation (7), and UMAP (8) are CPU-bound and
  effectively free on Modal by comparison.
- **All layers come from one forward pass.** Both extraction scripts
  (`04_extract_embeddings_evo2.py`, `extract_embeddings.py`) pass the full
  `layer_names` list into a single forward call, so sweeping 6 layers or all 32
  blocks costs the *same* GPU time — only more disk. **Layer count is not a cost
  driver.**
- **Two workloads:**
  - Task 2 (species): 10,000 fragments × **1,024 bp** — short, cheap.
  - Task 1 (regulatory): 10,000 windows × **10,000 bp** — long context,
    ~10× the tokens, the real cost center. The proposal itself flags
    "Evo 2 7B inference cost — budget compute now."
- **Models:** Task 1 → HyenaDNA, Caduceus, Evo 2 7B. Task 2 → NT v2 2.5B,
  DNABERT-S, Evo 2 7B.

## Assumed Modal GPU pricing

Modal bills per-second. Prices change — **verify against Modal's pricing page
before relying on these.**

| GPU | ~$/hr |
|---|---|
| L4 24 GB | 0.80 |
| A100 40 GB | 2.10 |
| A100 80 GB | 2.50 |
| H100 80 GB | 3.95 |

## Single clean pass

| Task | Model | GPU | Est. GPU-hr | Cost |
|---|---|---|---:|---:|
| 2 species (10k × 1kb) | Evo 2 7B | H100 | 1.5 | $5.9 |
| 2 species | NT v2 2.5B | A100-80 | 0.6 | $1.5 |
| 2 species | DNABERT-S | L4 | 0.4 | $0.3 |
| 1 regulatory (10k × 10kb) | Evo 2 7B | H100 | ~10 | $39.5 |
| 1 regulatory | HyenaDNA 1M | A100-80 | 1.5 | $3.8 |
| 1 regulatory | Caduceus | A100-40 | 1.0 | $2.1 |
| **Subtotal** | | | **~15 hr** | **~$53** |

**Evo 2 on the 10 kb Task 1 windows is ~75% of the bill and by far the biggest
uncertainty** — it could swing 2–4× (that one line alone could be $20–80)
depending on real throughput at 10 kb.

## Realistic total (with iteration)

The progress log already shows reality: layers got deleted and need
re-extraction, a fragment was dropped, runs get re-done. Apply a ~2.5× factor
for debugging, OOM re-runs, and re-extraction:

- **Clean pass: ~$50–60**
- **Realistic end-to-end: ~$130–200**

CPU work (probes, baselines, UMAP), storage (a few GB of HDF5), and egress add
only a few dollars.

## Modal-specific cost notes

- **Cache weights in a Modal Volume.** Evo 2 7B downloads ~14 GB; without a
  Volume you pay GPU time for that download on every cold start. NT v2 2.5B is
  similar.
- **Batch the short Task 2 fragments.** The current Evo 2 script is
  one-sequence-at-a-time, which wastes GPU seconds you are billed for. Batching
  cuts the species runs meaningfully.
- **Debug before scaling.** Don't run Evo 2 10 kb windows on H100 until the
  pipeline is validated on a few hundred windows first — that is where the money
  goes.
- Modal's Starter plan includes ~$30/mo of free credits, which covers most of a
  clean pass.

## Bottom line

Budget **~$150** for the full multi-model run with normal iteration. The clean
compute floor is **~$50**, dominated by Evo 2 on the long regulatory windows.
