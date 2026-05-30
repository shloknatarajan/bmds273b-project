"""Tests for the regulatory window-building logic.

Focus: `center_window`, the fix for the zero-regulatory-windows bug. The old
`tile_intervals` skipped any interval shorter than `window_size`, so every
cCRE (all <=350 bp) produced no window. Windows must instead be *centered* on
each labeled site and clamped to chromosome bounds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "regulatory"))

import pandas as pd  # noqa: E402
from pyfaidx import Fasta  # noqa: E402

from build_windows import (  # noqa: E402
    build_candidates,
    center_window,
    gc_matched_sample,
    materialize,
)


WIN = 10_000
CHROM = 1_000_000


def test_small_site_yields_full_window():
    # A 350 bp cCRE — the case the old code dropped entirely.
    start, end = center_window(500_000, 500_350, WIN, CHROM)
    assert end - start == WIN


def test_window_is_centered_on_site_midpoint():
    site_start, site_end = 500_000, 500_350
    start, end = center_window(site_start, site_end, WIN, CHROM)
    site_mid = (site_start + site_end) // 2
    win_mid = (start + end) // 2
    assert abs(win_mid - site_mid) <= 1
    # site fully contained
    assert start <= site_start and site_end <= end


def test_clamps_at_chromosome_start():
    # Site near position 0 cannot be centered without going negative.
    start, end = center_window(100, 450, WIN, CHROM)
    assert start == 0
    assert end == WIN


def test_clamps_at_chromosome_end():
    # Site near the end cannot be centered without exceeding chrom_len.
    start, end = center_window(CHROM - 200, CHROM - 50, WIN, CHROM)
    assert end == CHROM
    assert start == CHROM - WIN


def test_returns_none_when_chromosome_shorter_than_window():
    assert center_window(100, 200, WIN, chrom_len=5_000) is None


# --- build_candidates: coords-only, deduped, before any sequence fetch ---

CHROM_LENS = {"chr1": 1_000_000, "chr2": 800_000}


def _df(rows):
    return pd.DataFrame(rows, columns=["Chromosome", "Start", "End"])


def test_build_candidates_one_window_per_distinct_site():
    df = _df([("chr1", 100_000, 100_350), ("chr1", 500_000, 500_350)])
    cands = build_candidates(df, "regulatory", CHROM_LENS, WIN)
    assert len(cands) == 2
    assert all(c["label"] == "regulatory" for c in cands)
    assert all(c["end"] - c["start"] == WIN for c in cands)


def test_build_candidates_dedups_windows_clamped_to_same_start():
    # Two distinct sites near the chromosome start both clamp to [0, WIN].
    df = _df([("chr1", 100, 450), ("chr1", 200, 600)])
    cands = build_candidates(df, "regulatory", CHROM_LENS, WIN)
    assert len(cands) == 1
    assert cands[0]["start"] == 0 and cands[0]["end"] == WIN


def test_build_candidates_skips_unknown_chromosome():
    df = _df([("chrUn_xyz", 500_000, 500_350)])
    assert build_candidates(df, "gene_body", CHROM_LENS, WIN) == []


def test_build_candidates_skips_chromosome_shorter_than_window():
    short = {"chrTiny": 5_000}
    df = _df([("chrTiny", 100, 200)])
    assert build_candidates(df, "gene_body", short, WIN) == []


# --- end-to-end wiring against a tiny synthetic genome ---

def _write_fasta(path, contigs):
    with open(path, "w") as fh:
        for name, seq in contigs.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")


def test_candidates_to_materialize_to_gc_match(tmp_path):
    # 60 kb contig: GC-rich first half, AT-rich second half so both classes
    # populate distinct GC bins and matching has something to balance.
    seq = ("GC" * 15_000) + ("AT" * 15_000)
    fasta_path = tmp_path / "mini.fa"
    _write_fasta(fasta_path, {"chr1": seq})
    genome = Fasta(str(fasta_path), as_raw=False, sequence_always_upper=True)
    chrom_lens = {"chr1": len(seq)}

    gene_df = _df([("chr1", 8_000, 8_300), ("chr1", 40_000, 40_300)])
    reg_df = _df([("chr1", 12_000, 12_300), ("chr1", 45_000, 45_300)])

    gene_cands = build_candidates(gene_df, "gene_body", chrom_lens, WIN)
    reg_cands = build_candidates(reg_df, "regulatory", chrom_lens, WIN)

    gene_recs = materialize(gene_cands, genome, max_n_frac=0.05)
    reg_recs = materialize(reg_cands, genome, max_n_frac=0.05)

    # Every materialized window has a full-length sequence and a GC value.
    assert gene_recs and reg_recs
    for r in gene_recs + reg_recs:
        assert len(r["seq"]) == WIN
        assert 0.0 <= r["gc_content"] <= 1.0

    matched = gc_matched_sample(gene_recs, reg_recs, n_windows=10, gc_bins=10, seed=0)
    n_gene = sum(1 for r in matched if r["label"] == "gene_body")
    n_reg = sum(1 for r in matched if r["label"] == "regulatory")
    # GC matching yields equal counts per class.
    assert n_gene == n_reg
    assert n_gene >= 1
