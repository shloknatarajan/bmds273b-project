"""
Build gene_body vs regulatory labeled windows from GENCODE, ENCODE cCREs, and hg38.

    gene_body  : gene intervals with no cCRE overlap
    regulatory : cCREs not overlapping any exon (intronic/intergenic only)

Splits by chromosome: train chr1-18, val chr19-20, test chr21-22/X/Y.
Windows are GC-matched across classes to prevent the probe from learning composition.

Usage:
    python 01_build_windows.py \
        --gencode_gtf  data/regulatory/gencode.v47.annotation.gtf.gz \
        --encode_bed   data/regulatory/GRCh38-cCREs.bed \
        --genome_fasta data/regulatory/hg38.fa
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr
from pyfaidx import Fasta


# Chromosome-level split assignments

TRAIN_CHROMS = {f"chr{i}" for i in range(1, 19)}
VAL_CHROMS   = {"chr19", "chr20"}
TEST_CHROMS  = {"chr21", "chr22", "chrX", "chrY"}


def assign_split(chrom: str) -> str:
    c = chrom if chrom.startswith("chr") else f"chr{chrom}"
    if c in TRAIN_CHROMS:
        return "train"
    if c in VAL_CHROMS:
        return "val"
    if c in TEST_CHROMS:
        return "test"
    return "other"


# 1. Load annotations
def load_gencode(gtf_path: str) -> tuple[pr.PyRanges, pr.PyRanges]:
    """
    Returns (genes_pr, exons_pr) from a GENCODE GTF.
    genes_pr : feature == 'gene' records
    exons_pr : feature == 'exon' records
    """
    print(f"Loading GENCODE GTF: {gtf_path}")
    gtf = pr.read_gtf(gtf_path, as_df=False)

    genes = gtf[gtf.Feature == "gene"]
    exons = gtf[gtf.Feature == "exon"]

    print(f"  {len(genes)} gene records, {len(exons)} exon records")
    return genes, exons


def load_encode_ccres(bed_path: str) -> pr.PyRanges:
    """
    Loads the ENCODE cCRE BED file.
    The V4 BED9+ format has: chrom, start, end, accession, score,
    strand, thickStart, thickEnd, rgb [, cCRE_class].
    Only the first three columns are required by pyranges.
    """
    print(f"Loading ENCODE cCREs: {bed_path}")
    ccres = pr.read_bed(bed_path, as_df=False)
    print(f"  {len(ccres)} cCRE records")
    return ccres


# 2. Build labeled intervals

def build_labeled_intervals(
    genes: pr.PyRanges,
    exons: pr.PyRanges,
    ccres: pr.PyRanges,
) -> tuple[pr.PyRanges, pr.PyRanges]:
    """
    gene_body  = gene intervals - any overlap with a cCRE
    regulatory = cCRE intervals - any overlap with an exon
                 (keeps only intronic / intergenic cCREs)
    """
    print("Building gene_body intervals (genes - cCREs) ")
    gene_body = genes.subtract(ccres)

    print("Building regulatory intervals (cCREs - exons) ")
    regulatory = ccres.subtract(exons)

    print(f"  gene_body: {len(gene_body)} intervals")
    print(f"  regulatory: {len(regulatory)} intervals")
    return gene_body, regulatory


# 3. Tile intervals and extract sequences

def gc_fraction(seq: str) -> float:
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq) if seq else 0.0


def _fetch_seq(genome: Fasta, chrom: str, start: int, end: int) -> str | None:
    for key in (chrom, chrom.lstrip("chr"), f"chr{chrom.lstrip('chr')}"):
        if key in genome:
            try:
                seq = str(genome[key][start:end].seq).upper()
                return seq if len(seq) == (end - start) else None
            except Exception:
                continue
    return None


def tile_intervals(
    intervals: pr.PyRanges,
    window_size: int,
    label: str,
    genome: Fasta,
    max_n_frac: float,
) -> list[dict]:
    """
    Tiles all intervals into non-overlapping windows of `window_size` bp,
    extracts sequence, filters by N content, and computes GC fraction.
    """
    records = []
    df = intervals.as_df()

    for _, row in df.iterrows():
        chrom = str(row["Chromosome"])
        start = int(row["Start"])
        end   = int(row["End"])

        if (end - start) < window_size:
            continue

        for win_start in range(start, end - window_size + 1, window_size):
            win_end = win_start + window_size
            seq = _fetch_seq(genome, chrom, win_start, win_end)
            if seq is None:
                continue

            n_frac = seq.count("N") / len(seq)
            if n_frac > max_n_frac:
                continue

            records.append({
                "chrom":      chrom,
                "start":      win_start,
                "end":        win_end,
                "label":      label,
                "gc_content": gc_fraction(seq),
                "seq":        seq,
            })

    return records

# 4. GC-matched sampling
def gc_matched_sample(
    gene_records: list[dict],
    reg_records:  list[dict],
    n_windows:    int,
    gc_bins:      int = 10,
    seed:         int = 42,
) -> list[dict]:

    rng = random.Random(seed)
    bin_edges = np.linspace(0.0, 1.0, gc_bins + 1)

    def assign_bin(r: dict) -> int:
        return int(np.digitize(r["gc_content"], bin_edges[1:-1]))

    gene_by_bin: dict[int, list] = defaultdict(list)
    reg_by_bin:  dict[int, list] = defaultdict(list)

    for r in gene_records:
        gene_by_bin[assign_bin(r)].append(r)
    for r in reg_records:
        reg_by_bin[assign_bin(r)].append(r)

    per_bin_limit = max(1, n_windows // gc_bins)
    sampled_gene, sampled_reg = [], []

    for bin_id in range(gc_bins):
        g = gene_by_bin.get(bin_id, [])
        r = reg_by_bin.get(bin_id, [])
        rng.shuffle(g)
        rng.shuffle(r)
        n = min(len(g), len(r), per_bin_limit)
        sampled_gene.extend(g[:n])
        sampled_reg.extend(r[:n])

    # Final cap to n_windows per class
    rng.shuffle(sampled_gene)
    rng.shuffle(sampled_reg)
    return sampled_gene[:n_windows] + sampled_reg[:n_windows]


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gencode_gtf",  required=True,
                        help="GENCODE GTF file (plain or .gz)")
    parser.add_argument("--encode_bed",   required=True,
                        help="ENCODE cCRE BED file")
    parser.add_argument("--genome_fasta", required=True,
                        help="hg38 FASTA (pyfaidx-indexed; run `pyfaidx hg38.fa` first)")
    parser.add_argument("--out_dir",      default="data/regulatory")
    parser.add_argument("--window_size",  type=int, default=10_000,
                        help="Window size in bp (proposal requires >=10 kb)")
    parser.add_argument("--n_windows",    type=int, default=5_000,
                        help="Max windows per class after GC matching")
    parser.add_argument("--gc_bins",      type=int, default=10,
                        help="Number of GC-content bins for matching")
    parser.add_argument("--max_n_frac",   type=float, default=0.05,
                        help="Max fraction of ambiguous (N) bases per window")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load annotations ---
    genes, exons = load_gencode(args.gencode_gtf)
    ccres  = load_encode_ccres(args.encode_bed)

    # --- Build labeled intervals ---
    print("\nBuilding labeled intervals ...")
    gene_body_pr, regulatory_pr = build_labeled_intervals(genes, exons, ccres)

    # --- Open genome ---
    print(f"\nOpening genome FASTA: {args.genome_fasta}")
    genome = Fasta(args.genome_fasta, as_raw=False, sequence_always_upper=True)

    # --- Tile and extract sequences ---
    print(f"\nTiling gene_body intervals (window_size={args.window_size:,} bp) ...")
    gene_windows = tile_intervals(
        gene_body_pr, args.window_size, "gene_body", genome, args.max_n_frac
    )
    print(f"  {len(gene_windows):,} candidate gene_body windows")

    print("Tiling regulatory intervals ...")
    reg_windows = tile_intervals(
        regulatory_pr, args.window_size, "regulatory", genome, args.max_n_frac
    )
    print(f"  {len(reg_windows):,} candidate regulatory windows")

    # --- GC-matched sampling ---
    print(f"\nGC-matched sampling "
          f"(target={args.n_windows}/class, {args.gc_bins} bins) ...")
    matched = gc_matched_sample(
        gene_windows, reg_windows, args.n_windows, args.gc_bins, args.seed
    )
    n_gene = sum(1 for r in matched if r["label"] == "gene_body")
    n_reg  = sum(1 for r in matched if r["label"] == "regulatory")
    print(f"  {n_gene:,} gene_body windows retained")
    print(f"  {n_reg:,} regulatory windows retained")

    # --- Assign IDs and splits ---
    records = []
    for r in matched:
        window_id = f"{r['label']}_{r['chrom']}_{r['start']}"
        records.append({
            "window_id":  window_id,
            "chrom":      r["chrom"],
            "start":      r["start"],
            "end":        r["end"],
            "label":      r["label"],
            "split":      assign_split(r["chrom"]),
            "gc_content": round(r["gc_content"], 4),
            "seq":        r["seq"],
        })

    df = pd.DataFrame(records)

    # Save windows TSV
    out_tsv = out_dir / "windows.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nSaved {len(df):,} windows to {out_tsv}")

    #  Print and save split stats
    lines = [
        f"Total windows : {len(df):,}",
        f"  gene_body   : {(df['label']=='gene_body').sum():,}",
        f"  regulatory  : {(df['label']=='regulatory').sum():,}",
        "",
        "Split breakdown:",
    ]
    for split in ["train", "val", "test", "other"]:
        sub = df[df["split"] == split]
        g   = (sub["label"] == "gene_body").sum()
        r   = (sub["label"] == "regulatory").sum()
        lines.append(f"  {split:5s}: {len(sub):5,} total  ({g:,} gene_body / {r:,} regulatory)")

    stats_text = "\n".join(lines)
    print("\n" + stats_text)
    (out_dir / "split_stats.txt").write_text(stats_text + "\n")


if __name__ == "__main__":
    main()
