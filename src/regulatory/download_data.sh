#!/usr/bin/env bash
# ============================================================
# 00_download_data.sh
# Downloads all required data for the regulatory vs. gene-body
# classification task (Task 1).
#
# Sources
#   GENCODE v47  — human gene / exon annotations (GTF)
#   ENCODE V4    — GRCh38 candidate cis-regulatory elements (BED)
#   hg38         — GRCh38 reference genome FASTA (UCSC)
#
# Expected disk: ~3.5 GB (hg38 FASTA dominates).
# ============================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data/regulatory}"
mkdir -p "$DATA_DIR"

# ----------------------------------------------------------------
# 1. GENCODE v47 comprehensive gene annotation
# ----------------------------------------------------------------
echo "=== [1/3] GENCODE v47 annotation GTF ==="
wget -nc -P "$DATA_DIR" \
  "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.annotation.gtf.gz"
echo "    -> $DATA_DIR/gencode.v47.annotation.gtf.gz  (kept compressed; pyranges reads .gz)"

# ----------------------------------------------------------------
# 2. ENCODE GRCh38 cCRE Registry V4
# ----------------------------------------------------------------
echo ""
echo "=== [2/3] ENCODE GRCh38 cCRE registry V4 ==="
# Primary: SCREEN (Weng lab) Registry-V4 direct download
CCRE_URL="https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed"
CCRE_OUT="$DATA_DIR/GRCh38-cCREs.bed"

if [ ! -f "$CCRE_OUT" ]; then
  echo "  Trying SCREEN Registry-V4 download ..."
  wget -O "$CCRE_OUT" "$CCRE_URL" || {
    echo "  Primary URL failed. Trying ENCODE portal ..."
    # Fallback: ENCODE portal file ENCFF420VPZ (V4 GRCh38 cCREs, BED)
    wget -O "${CCRE_OUT}.gz" \
      "https://www.encodeproject.org/files/ENCFF420VPZ/@@download/ENCFF420VPZ.bed.gz"
    gunzip -k "${CCRE_OUT}.gz"
  }
else
  echo "  $CCRE_OUT already exists, skipping."
fi
echo "    -> $CCRE_OUT"

# ----------------------------------------------------------------
# 3. hg38 / GRCh38 reference FASTA
# ----------------------------------------------------------------
echo ""
echo "=== [3/3] hg38 reference genome FASTA (~900 MB compressed) ==="
HG38_GZ="$DATA_DIR/hg38.fa.gz"
HG38_FA="$DATA_DIR/hg38.fa"

if [ ! -f "$HG38_FA" ]; then
  if [ ! -f "$HG38_GZ" ]; then
    wget -nc -P "$DATA_DIR" \
      "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
  fi
  echo "  Decompressing hg38.fa.gz (~3 GB uncompressed, takes a few minutes) ..."
  gunzip -k "$HG38_GZ"
else
  echo "  $HG38_FA already exists, skipping decompress."
fi

echo ""
echo "  Indexing hg38.fa with pyfaidx ..."
python -c "from pyfaidx import Fasta; Fasta('$HG38_FA'); print('  Index OK')"
echo "    -> $HG38_FA  (+ $HG38_FA.fai)"

# ----------------------------------------------------------------
# Done
# ----------------------------------------------------------------
echo ""
echo "=== Done — run build_windows.py next ==="
echo ""
echo "  python src/regulatory/build_windows.py \\"
echo "      --gencode_gtf  $DATA_DIR/gencode.v47.annotation.gtf.gz \\"
echo "      --encode_bed   $DATA_DIR/GRCh38-cCREs.bed \\"
echo "      --genome_fasta $DATA_DIR/hg38.fa \\"
echo "      --out_dir      $DATA_DIR"
