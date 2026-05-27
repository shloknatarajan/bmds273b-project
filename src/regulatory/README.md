# Regulatory vs. Gene Body Classification (Task 1)

Probes frozen DNA language model embeddings to classify 10 kb genomic
windows as **gene body** or **regulatory element**, using annotations
from GENCODE v47 and the ENCODE cCRE Registry V4.

## Pipeline

| Script | Description |
|--------|-------------|
| `download_data.sh` | Download GENCODE GTF, ENCODE cCREs, hg38 FASTA |
| `build_windows.py` | Build GC-matched gene_body / regulatory windows → `windows.tsv` |
| `src/extract_embeddings.py` | Extract per-layer embeddings from frozen models (shared) |
| `train_probes.py` | Train logistic / MLP probes, layer-wise evaluation |
| `umap_visualization.py` | UMAP projections colored by label / chromosome |
| `baselines.py` | k-mer TF-IDF and GC+dinucleotide baselines |

## How windows.tsv flows through the pipeline

`windows.tsv` is the single artifact produced by `build_windows.py` that
all downstream scripts consume. Each row is one 10 kb genomic window and
contains everything needed for the rest of the pipeline:

| Column | Used by | Purpose |
|--------|---------|---------|
| `window_id` | extract_embeddings | Row identity; written to `window_ids.txt` so embeddings can be matched back |
| `seq` | extract_embeddings, baselines | Raw nucleotide sequence passed to model tokenizer (embeddings) or k-mer featurizer (baselines) |
| `label` | train_probes, baselines, umap | Classification target: `gene_body` or `regulatory` |
| `split` | train_probes, baselines, umap | Chromosome-level assignment (`train`/`val`/`test`); row order in `windows.tsv` maps directly to row order in the HDF5 embedding files, so slicing by split index works without joins |
| `gc_content` | (audit) | Confirms GC matching held; not used as a feature |
| `chrom` | umap | Coloring UMAP plots by chromosome |
| `start` / `end` | (audit) | Genomic coordinates for tracing back to annotation |

The key design constraint is that **row order is preserved end-to-end**.
`extract_embeddings.py` iterates `windows.tsv` sequentially and writes
embeddings at the same row index into `layer_XX.h5`. So when `train_probes.py`
loads a split, it finds the matching embedding rows by boolean-indexing the
same DataFrame — no merge key needed, just `df["split"] == "train"` → row
indices → HDF5 slice.

## Models

| Model | Role | Context | Notes |
|-------|------|---------|-------|
| HyenaDNA | primary | up to 1 M nt | Long Hyena conv, single-nucleotide tokens |
| Caduceus | primary | up to 131 k nt | Bidirectional Mamba, RC-equivariant |
| Evo 2 7B | primary | 8 192 tokens (HF) | StripedHyena, multi-kingdom pre-training |
| NT v2 2.5B | baseline | 2 048 tokens | Short-context transformer comparison |
| DNABERT-S | baseline | 512 tokens | Short-context transformer comparison |

## Key design choices

- **Chromosome-level split** — train chr1–18, val chr19–20, test chr21–22/X/Y.
  Random splits leak signal via linkage disequilibrium.
- **GC-content matching** — windows sampled in matched GC bins so probes
  cannot exploit composition differences between classes.
- **N-content filter** — windows with >5 % ambiguous bases excluded.
- **Window size** — 10 kb; long-context models use the full sequence,
  short-context models truncate.
- **Label definitions**
  - `gene_body`  : GENCODE gene interval *minus* any cCRE overlap
  - `regulatory` : ENCODE cCRE *minus* any exon overlap (intronic / intergenic)

## Quick start

```bash
# 1. Download data (~3.5 GB)
DATA_DIR=data/regulatory bash src/regulatory/download_data.sh

# 2. Build windows
python src/regulatory/build_windows.py \
    --gencode_gtf  data/regulatory/gencode.v47.annotation.gtf.gz \
    --encode_bed   data/regulatory/GRCh38-cCREs.bed \
    --genome_fasta data/regulatory/hg38.fa \
    --out_dir      data/regulatory

# 3. Extract embeddings for each model (needs GPU)
python src/extract_embeddings.py \
    --model hyenadna \
    --windows data/regulatory/windows.tsv \
    --out_dir data/regulatory/embeddings/hyenadna \
    --device cuda --batch_size 2

python src/extract_embeddings.py \
    --model caduceus \
    --windows data/regulatory/windows.tsv \
    --out_dir data/regulatory/embeddings/caduceus \
    --device cuda --batch_size 2

python src/extract_embeddings.py \
    --model evo2 \
    --windows data/regulatory/windows.tsv \
    --out_dir data/regulatory/embeddings/evo2 \
    --device cuda --batch_size 1

# 4. Train probes
python src/regulatory/train_probes.py

# 5. Sequence baselines (CPU, fast)
python src/regulatory/baselines.py

# 6. UMAP (final layer, test split)
python src/regulatory/umap_visualization.py
```

## Outputs

```
data/regulatory/
  windows.tsv                       window sequences + labels + splits
  split_stats.txt                   window counts per split
  embeddings/
    hyenadna/layer_00.h5 ...        (N_windows, hidden_dim) float32
    caduceus/layer_00.h5 ...
    evo2/layer_00.h5 ...
  results/
    probe_results.tsv               (model, layer, probe, f1, auc, auprc)
    baseline_results.tsv            sequence-feature baseline scores
    layer_curves_<model>.png        metric vs layer per model
    model_comparison.png            best-layer AUPRC across models
    baseline_comparison.png         baseline bar chart
    umap/
      <model>_layer<idx>_label.png  UMAP colored by label
      <model>_layer<idx>_chrom.png  UMAP colored by chromosome
```
