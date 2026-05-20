#!/usr/bin/env bash
# ============================================================
# 00_download_data.sh
# Downloads all required data for the Evo 2 7B species
# prediction task. Run once before the Python pipeline.
# Expected total disk: ~50–100 GB depending on species count. (?? only will take up a few GB in storage in reality)
# ============================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"/{gtdb,ncbi_genomes,trees}

echo "=== [1/4] Downloading GTDB r232 taxonomy files ==="
# Bacterial taxonomy + archaeal taxonomy
wget -nc -P "$DATA_DIR/gtdb" \
  https://data.gtdb.ecogenomic.org/releases/release232/232.0/bac120_taxonomy_r232.tsv.gz \
  https://data.gtdb.ecogenomic.org/releases/release232/232.0/ar53_taxonomy_r232.tsv.gz

# Species-level representative genome lists (much smaller than full 900k)
wget -nc -P "$DATA_DIR/gtdb" \
  https://data.gtdb.ecogenomic.org/releases/release232/232.0/bac120_metadata_r232.tsv.gz \
  https://data.gtdb.ecogenomic.org/releases/release232/232.0/ar53_metadata_r232.tsv.gz

# Newick trees for phylogenetic distance computation
wget -nc -P "$DATA_DIR/trees" \
  https://data.gtdb.ecogenomic.org/releases/release232/232.0/bac120_r232.tree.gz \
  https://data.gtdb.ecogenomic.org/releases/release232/232.0/ar53_r232.tree.gz

gunzip -kf "$DATA_DIR/gtdb/"*.gz 2>/dev/null || true
gunzip -kf "$DATA_DIR/trees/"*.gz 2>/dev/null || true

echo "=== [2/4] Installing NCBI Datasets CLI ==="
# https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/
if ! command -v datasets &>/dev/null; then
  curl -o "$DATA_DIR/datasets" \
    "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets"
  chmod +x "$DATA_DIR/datasets"
  export PATH="$DATA_DIR:$PATH"
  echo "Add $DATA_DIR to your PATH or copy datasets to /usr/local/bin"
fi

echo "=== [3/4] Subsampled genome list will be built by 01_subsample_species.py ==="
echo "    That script reads GTDB metadata and outputs accessions_to_download.txt"

echo "=== [4/4] Done — run 01_subsample_species.py next ==="
echo "    Data directory: $DATA_DIR"
