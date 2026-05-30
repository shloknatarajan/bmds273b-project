"""
02_extract_fragments.py
-----------------------
For each downloaded genome, extracts random non-overlapping fragments
of a fixed length (default 1 kb — Evo 2 can handle much longer, but
1 kb balances coverage and RAM within the 46 GB GPU constraint).

Outputs
-------
data/fragments/
    fragments.tsv     — columns: frag_id, accession, species, phylum,
                        domain, contig, start, end, seq
    species_split.json — maps each species to "train" / "val" / "test"

Key design choices (per technical_steps.md)
-------------------------------------------
- Species-level train/val/test split (not fragment-level) to avoid
  inflated metrics from the model having seen the same genome.
- Skip windows with >5% ambiguous bases (Ns).
- Fragments sampled from non-overlapping, randomly-shuffled windows.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from Bio import SeqIO


# ---------------------------------------------------------------------------
# Fragment extraction helpers
# ---------------------------------------------------------------------------

def extract_fragments_from_fasta(
    fasta_path: Path,
    n_fragments: int,
    frag_len: int,
    max_n_frac: float = 0.05,
    seed: int = 42,
) -> list[dict]:
    """
    Reads a FASTA, tiles non-overlapping windows, shuffles, and returns
    up to `n_fragments` windows that pass the N-content filter.
    """
    rng = random.Random(seed)
    candidates = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        contig_len = len(seq)
        if contig_len < frag_len:
            continue

        # Tile the contig into non-overlapping windows
        starts = list(range(0, contig_len - frag_len + 1, frag_len))
        rng.shuffle(starts)

        for start in starts:
            end = start + frag_len
            subseq = seq[start:end]
            n_frac = subseq.count("N") / frag_len
            if n_frac <= max_n_frac:
                candidates.append({
                    "contig": record.id,
                    "start":  start,
                    "end":    end,
                    "seq":    subseq,
                })
            if len(candidates) >= n_fragments * 3:   # over-sample, then trim
                break
        if len(candidates) >= n_fragments * 3:
            break

    rng.shuffle(candidates)
    return candidates[:n_fragments]


def find_fasta(genome_dir: Path) -> Path | None:
    """Finds the primary FASTA inside an NCBI Datasets genome directory."""
    for pattern in ["*.fna", "*.fa", "*.fasta", "*.fna.gz"]:
        hits = list(genome_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------------
# Species-level train / val / test split
# ---------------------------------------------------------------------------

def split_species(
    species_index: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac:   float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """
    Returns a dict mapping species name → split label.
    Stratified by phylum so each split has proportional phylum coverage.
    """
    rng = random.Random(seed)
    split_map: dict[str, str] = {}

    for phylum, grp in species_index.groupby("phylum"):
        species_list = grp["species"].unique().tolist()
        rng.shuffle(species_list)
        n = len(species_list)
        n_train = max(1, int(n * train_frac))
        n_val   = max(1, int(n * val_frac))

        for sp in species_list[:n_train]:
            split_map[sp] = "train"
        for sp in species_list[n_train : n_train + n_val]:
            split_map[sp] = "val"
        for sp in species_list[n_train + n_val :]:
            split_map[sp] = "test"

    return split_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species_index",    default="data/species_index.tsv")
    parser.add_argument("--genome_dir",       default="data/ncbi_genomes")
    parser.add_argument("--out_dir",          default="data/fragments")
    parser.add_argument("--frag_len",         type=int, default=1024,
                        help="Fragment length in bp. Evo 2 can handle up to 8192; "
                             "1024 is recommended given the 46 GB GPU RAM budget.")
    parser.add_argument("--frags_per_genome", type=int, default=50,
                        help="Max fragments to extract per genome.")
    parser.add_argument("--max_n_frac",       type=float, default=0.05)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.species_index, sep="\t")
    genome_dir = Path(args.genome_dir)

    print(f"Building fragments (len={args.frag_len} bp, max {args.frags_per_genome}/genome) ...")

    # --- Species split ---
    split_map = split_species(index, seed=args.seed)
    with open(out_dir / "species_split.json", "w") as f:
        json.dump(split_map, f, indent=2)
    print(f"Split: {sum(1 for v in split_map.values() if v=='train')} train / "
          f"{sum(1 for v in split_map.values() if v=='val')} val / "
          f"{sum(1 for v in split_map.values() if v=='test')} test species")

    # --- Fragment extraction ---
    records = []
    missing = []

    for _, row in index.iterrows():
        acc     = row["accession"]
        species = row["species"]
        phylum  = row["phylum"]
        domain  = row["domain"]

        clean_acc = acc.replace("GB_", "").replace("RS_", "")
        acc_dir = genome_dir / clean_acc
        fasta   = find_fasta(acc_dir) if acc_dir.exists() else None

        if fasta is None:
            missing.append(acc)
            continue

        frags = extract_fragments_from_fasta(
            fasta,
            n_fragments=args.frags_per_genome,
            frag_len=args.frag_len,
            max_n_frac=args.max_n_frac,
            seed=args.seed,
        )

        split = split_map.get(species, "test")
        for i, frag in enumerate(frags):
            frag_id = f"{acc}_{frag['contig']}_{frag['start']}"
            records.append({
                "frag_id":  frag_id,
                "accession": acc,
                "species":  species,
                "phylum":   phylum,
                "domain":   domain,
                "split":    split,
                "contig":   frag["contig"],
                "start":    frag["start"],
                "end":      frag["end"],
                "seq":      frag["seq"],
            })

    if missing:
        print(f"WARNING: {len(missing)} accessions had no FASTA: {missing[:5]} ...")

    df = pd.DataFrame(records)
    out_tsv = out_dir / "fragments.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nExtracted {len(df)} fragments from {df['accession'].nunique()} genomes")
    print(f"Split sizes (fragments): {df['split'].value_counts().to_dict()}")
    print(f"Saved to {out_tsv}")


if __name__ == "__main__":
    main()
