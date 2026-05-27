# Task 1 — Regulatory vs Gene Body

Layer-wise probing of frozen DNA language models on a **gene_body vs
regulatory** binary task. Mirrors the `src/species/` pipeline (Task 2).

See `docs/technical_steps.md` §2, §4–§8.

## Pipeline

| Step | Script | What it does | Needs GPU? |
|---|---|---|---|
| 00 | `00_download_data.sh` | Download GENCODE GTF, ENCODE cCRE BED, hg38 FASTA | no |
| 01 | `01_build_windows.py` | Build GC-matched gene_body/regulatory windows; chromosome-level split → `windows.tsv` | no |
| 02 | `../extract_embeddings.py` | Frozen-LM embeddings per layer (`layer_NN.h5` + `window_ids.txt`) for evo2 / hyenadna / caduceus / nt / dnabert_s | **yes** |
| 03 | `03_train_probes.py` | Logistic + MLP probes per (model, layer); F1/AUC/AUPRC; layer curves | no |
| 04 | `04_baselines.py` | k-mer TF-IDF + logistic, and shallow 1D CNN baselines | CNN benefits |
| 05 | `05_umap_visualization.py` | UMAP colored by class and by GC content | no |

## Labels

- **gene_body**: GENCODE gene intervals minus any cCRE overlap.
- **regulatory**: ENCODE cCREs minus exon overlap (intronic/intergenic only).

Classes are **GC-matched** (step 01) so probes can't win on composition
alone. Split is **by chromosome** (train chr1–18 / val chr19–20 /
test chr21,22,X,Y) to avoid linkage leakage.

## Quick start

```bash
# 1. data (~4 GB)
bash src/regulatory/00_download_data.sh

# 2. windows
python src/regulatory/01_build_windows.py \
    --gencode_gtf  data/regulatory/gencode.v47.annotation.gtf.gz \
    --encode_bed   data/regulatory/GRCh38-cCREs.bed \
    --genome_fasta data/regulatory/hg38.fa

# 3. embeddings (GPU) — repeat per model
python src/extract_embeddings.py --model nt \
    --windows data/regulatory/windows.tsv \
    --out_dir data/embeddings/nt --device cuda

# 4. probes + baselines + UMAP (CPU ok)
python src/regulatory/03_train_probes.py
python src/regulatory/04_baselines.py
python src/regulatory/05_umap_visualization.py --embeddings_dir data/embeddings/nt
```

## Notes / dependencies

- `05_umap_visualization.py` needs **`umap-learn`**, which is *not yet
  pinned in `pyproject.toml`* — add it before running (also used by
  `src/species/06`).
- `03`/`05` align embedding rows to `windows.tsv` via `window_ids.txt`,
  so the H5 row order is robust even if regenerated separately.
- Embedding extraction is the only GPU step; see the Modal cost notes in
  the project log under `logs/`.
