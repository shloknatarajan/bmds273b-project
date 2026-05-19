# Technical Steps

Implementation outline for the layer-wise / phylogenetic probing project described in `project_proposal.md`.

## External data sources

| Source | Format | Shape / scale | Used for |
|---|---|---|---|
| **GENCODE** (human gene annotations, latest release) | GTF/GFF3 | Records of `chrom, start, end, strand, feature_type, gene_id, gene_type` — ~78k genes, ~1.6M transcripts | Task 1 — "gene body" label |
| **ENCODE cCRE registry** (V4) | BED9+ | `chrom, start, end, accession, cCRE_class` (PLS / pELS / dELS / CTCF-only / DNase-H3K4me3) — ~1M elements | Task 1 — "regulatory" label |
| **GRCh38 / hg38** reference genome (UCSC or NCBI) | FASTA | 24 chromosomes, ~3.1 Gb total | Sequence extraction for Task 1 windows |
| **GTDB r232** | TSV taxonomy + Newick tree | ~901,341 genomes × `(accession, d__;p__;c__;o__;f__;g__;s__)`; tree gives pairwise phylogenetic distances | Task 2 — species labels + ground-truth evolutionary distance |
| **NCBI nucleotide / Datasets CLI** | FASTA (per assembly) | One multi-FASTA per genome; bacterial ~1–10 Mb, archaeal ~1–6 Mb | Task 2 — actual sequences for GTDB accessions |
| **NCBI RefSeq genomes** (optional eukaryote expansion) | FASTA + GFF | Per-species assemblies, MB–GB scale | Task 2 stretch goal |

GTDB lists ~900k genomes — subsample (e.g., one representative per species cluster, stratified across phyla) rather than pulling all of NCBI.

## Technical steps

### 1. Environment & infra
- Python env with `transformers`, `torch`, `peft`, `accelerate`, `scikit-learn`, `pyfaidx`/`pysam`, `pybedtools`/`pyranges`, `Biopython`, `ete3`/`dendropy` (tree parsing), `umap-learn`.
- GPU access — Evo 2 7B in fp16 needs ~16–20 GB just for weights; long-context inference (HyenaDNA up to ~1 M nt) needs A100 80 GB or H100. Plan for Stanford Sherlock or similar.
- Caching layout: `embeddings/{model}/{layer}/{window_id}.npy` (or a single memmap / HDF5 per model).

### 2. Task 1 — data pipeline (regulatory vs gene)
1. Download GENCODE GTF, ENCODE cCRE BED, GRCh38 FASTA.
2. Build interval sets with `pyranges`:
   - **Gene-body intervals**: GENCODE gene records *minus* any region overlapping a cCRE.
   - **Regulatory intervals**: cCREs falling inside introns or intergenic regions (subtract exons).
3. Sample matched ≥10 kb windows around each labeled site; match on **length** and **GC content** between classes (otherwise the probe learns composition, not biology).
4. Extract sequence with `pyfaidx`.
5. **Chromosome-level split** (e.g., train on chr1–18, val chr19–20, test chr21–22, X, Y) — random splits leak via linkage.

### 3. Task 2 — data pipeline (species)
1. Parse GTDB `bac120_taxonomy_r232.tsv` + `ar53_taxonomy_r232.tsv`; load species tree.
2. Stratified sample of species (e.g., balanced across phyla, then ~N genomes per species).
3. Pull FASTAs via `datasets download genome accession <GCA/GCF>` (NCBI Datasets CLI) or `efetch`.
4. From each genome, draw random fragments (lengths matched to model context: ~1 kb for NT v2 / DNABERT-S, up to context limit for Evo 2). Avoid Ns; ensure fragments come from non-overlapping windows.
5. Compute pairwise phylogenetic distances from the Newick tree once and cache as a square matrix indexed by species ID.
6. Split species (not fragments) into train/test so the model is evaluated on **held-out species**, not held-out fragments.

### 4. Model loading & frozen embedding extraction
For each model (HyenaDNA, Caduceus, Evo 2 7B, NT v2 2.5B, DNABERT-S):
- Load from HuggingFace with `output_hidden_states=True`, set `model.eval()` and `requires_grad=False`.
- For each window/fragment, forward pass → list of `(layer, seq_len, hidden_dim)` tensors.
- **Mean-pool over the sequence axis per layer** → one vector per (window, layer).
- Cache to disk; reuse across all downstream probes.

### 5. Probes
- Per `(model, layer)`: train logistic regression (L2-regularized) and a shallow MLP (1 hidden layer, dropout) on cached embeddings.
- Standardize features (StandardScaler) before logistic.
- Tune regularization on val split.

### 6. Baselines
- **k-mer TF–IDF** (k = 4–6) + logistic regression on raw sequence.
- **Shallow 1D CNN** on one-hot sequence, trained end-to-end on the same windows.

### 7. Evaluation
- **Task 1**: AUPRC / AUC / F1 per (model, layer); plot as layer-wise curves.
- **Task 2 — classification**: same metrics, multi-class or one-vs-rest.
- **Task 2 — embedding distance**: cosine + Euclidean pairwise distance between fragment embeddings, then **Spearman ρ** vs GTDB tree distance. Compute per layer.
- Significance: bootstrap CIs over windows/species.

### 8. Analysis & writeup
- Layer-wise probe curves per architecture.
- UMAP of embeddings colored by phylum / cCRE class.
- Compare long-context (HyenaDNA / Caduceus / Evo 2) vs short-context (NT v2 / DNABERT-S) on where regulatory signal emerges.

## Sharp edges to flag early
- **Evo 2 7B inference cost** — budget compute now; a single 1 Mb window across all layers is expensive.
- **GC/length matching for Task 1** — without it, results are uninterpretable.
- **Held-out *species*** for Task 2 — held-out fragments give inflated numbers because the model has seen the same genome.
- **Tokenizer differences** — NT uses 6-mer tokens, DNABERT-S uses BPE-like, HyenaDNA/Caduceus are single-nucleotide. Pooling is comparable but per-token "resolution" is not.
