#!/usr/bin/env bash
# ============================================================
# 00_download_data.sh
# Downloads all required data for Task 1 (regulatory vs gene
# body). Run once before 01_build_windows.py.
#
# Downloads (~4 GB total, mostly hg38):
#   - GENCODE human gene annotation (GTF)
#   - ENCODE SCREEN cCRE registry (BED, GRCh38)
#   - GRCh38 / hg38 reference genome (FASTA)
#
# URLs are overridable via env vars so you can pin a release.
# ============================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data/regulatory}"
mkdir -p "$DATA_DIR"

# --- Release / URL configuration (override via env) ---------
GENCODE_RELEASE="${GENCODE_RELEASE:-47}"
GENCODE_URL="${GENCODE_URL:-https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${GENCODE_RELEASE}/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz}"
# ENCODE SCREEN Registry of cCREs V4 (GRCh38). Verify the current
# path at https://screen.encodeproject.org / https://downloads.wenglab.org
CCRE_URL="${CCRE_URL:-https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.V4.bed}"
HG38_URL="${HG38_URL:-https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz}"

echo "=== [1/3] GENCODE annotation (release ${GENCODE_RELEASE}) ==="
wget -nc -P "$DATA_DIR" "$GENCODE_URL"

echo "=== [2/3] ENCODE cCRE registry ==="
# Save under a stable name regardless of the remote filename.
if [ ! -f "$DATA_DIR/GRCh38-cCREs.bed" ]; then
  wget -nc -O "$DATA_DIR/GRCh38-cCREs.bed" "$CCRE_URL"
fi

echo "=== [3/3] hg38 reference FASTA (~1 GB gz, ~3 GB unzipped) ==="
wget -nc -P "$DATA_DIR" "$HG38_URL"
if [ ! -f "$DATA_DIR/hg38.fa" ]; then
  echo "  Decompressing hg38.fa.gz ..."
  gunzip -kf "$DATA_DIR/hg38.fa.gz"
fi

echo "=== Building FASTA index (.fai) ==="
# pyfaidx builds the .fai on first open; do it now so build_windows is fast.
# Falls back silently if pyfaidx isn't importable yet.
python -c "from pyfaidx import Fasta; Fasta('$DATA_DIR/hg38.fa'); print('  .fai ready')" \
  2>/dev/null || echo "  (skipped — run after installing deps; pyfaidx will index on first use)"

cat <<EOF

Done. Files in $DATA_DIR:
  gencode.v${GENCODE_RELEASE}.annotation.gtf.gz
  GRCh38-cCREs.bed
  hg38.fa (+ hg38.fa.fai)

Next:
  python src/regulatory/01_build_windows.py \\
      --gencode_gtf  $DATA_DIR/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz \\
      --encode_bed   $DATA_DIR/GRCh38-cCREs.bed \\
      --genome_fasta $DATA_DIR/hg38.fa
EOF
