"""
01_subsample_species.py
-----------------------
Reads GTDB r232 metadata, subsamples species representatives
stratified by phylum, then downloads their FASTA files from NCBI.

Outputs
-------
data/species_index.tsv          — columns: accession, species, phylum, domain
data/accessions_to_download.txt — one GCA/GCF accession per line
data/ncbi_genomes/<accession>/  — one directory per genome with *.fna files

Usage
-----
  python 01_subsample_species.py \
      --gtdb_dir data/gtdb \
      --out_dir data \
      --n_phyla 20 \
      --species_per_phylum 10 \
      --datasets_bin data/datasets   # path to NCBI datasets CLI
"""

import argparse
import gzip
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Parse GTDB metadata to get one representative per species
# ---------------------------------------------------------------------------

def parse_gtdb_metadata(gtdb_dir: Path) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      accession, domain, phylum, class, order, family, genus, species
    filtered to GTDB species representatives (gtdb_representative == 't').
    """
    rows = []
    for fname, domain in [
        ("bac120_metadata_r232.tsv", "Bacteria"),
        ("ar53_metadata_r232.tsv",   "Archaea"),
    ]:
        fpath = gtdb_dir / fname
        if not fpath.exists():
            # try gzipped
            fpath = gtdb_dir / (fname + ".gz")
            opener = gzip.open
        else:
            opener = open

        with opener(fpath, "rt") as fh:
            df = pd.read_csv(fh, sep="\t", low_memory=False)

        # Keep only GTDB species representatives
        rep_col = "gtdb_representative"
        if rep_col in df.columns:
            df = df[df[rep_col] == "t"]

        # Parse lineage string: d__;p__;c__;o__;f__;g__;s__
        lineage_col = "gtdb_taxonomy"
        if lineage_col in df.columns:
            lineage_parsed = df[lineage_col].str.split(";", expand=True)
            lineage_parsed.columns = ["domain_t","phylum_t","class_t","order_t","family_t","genus_t","species_t"]
            for col in lineage_parsed.columns:
                lineage_parsed[col] = lineage_parsed[col].str.split("__").str[-1].str.strip()
            df = pd.concat([df[["accession"]], lineage_parsed], axis=1)
        else:
            raise ValueError(f"Column '{lineage_col}' not found in {fname}")

        df["domain"] = domain
        df = df.rename(columns={
            "phylum_t": "phylum",
            "class_t": "class_",
            "order_t": "order_",
            "family_t": "family",
            "genus_t": "genus",
            "species_t": "species",
        })
        rows.append(df[["accession","domain","phylum","class_","order_","family","genus","species"]])

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Stratified subsample: top N phyla × K species per phylum
# ---------------------------------------------------------------------------

def stratified_subsample(meta: pd.DataFrame, n_phyla: int, species_per_phylum: int,
                         seed: int = 42) -> pd.DataFrame:
    """
    Selects top `n_phyla` phyla by species count (to ensure enough representatives),
    then randomly samples `species_per_phylum` species per phylum.
    """
    phylum_counts = meta["phylum"].value_counts()
    top_phyla = phylum_counts[phylum_counts >= species_per_phylum].head(n_phyla).index.tolist()

    sampled = (
        meta[meta["phylum"].isin(top_phyla)]
        .groupby("phylum", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), species_per_phylum), random_state=seed))
    )
    print(f"Subsampled {len(sampled)} genomes across {sampled['phylum'].nunique()} phyla")
    print(sampled["phylum"].value_counts().to_string())
    return sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Download FASTAs via NCBI Datasets CLI
# ---------------------------------------------------------------------------

def download_genomes(accessions: list[str], out_dir: Path, datasets_bin: str) -> None:
    """
    Downloads genome FASTA for each accession using the NCBI Datasets CLI.
    Skips accessions that already have a downloaded directory.
    """
    genome_dir = out_dir / "ncbi_genomes"
    genome_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 50  # NCBI recommends batching
    todo = [a for a in accessions if not (genome_dir / a).exists()]
    print(f"Downloading {len(todo)} genomes (already have {len(accessions)-len(todo)})")

    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        batch_str = " ".join(batch)
        zip_path = genome_dir / f"batch_{i}.zip"

        cmd = (
            f"{datasets_bin} download genome accession {batch_str} "
            f"--include genome "
            f"--filename {zip_path}"
        )
        print(f"  Batch {i//batch_size + 1}: {cmd[:120]}...")
        subprocess.run(cmd, shell=True, check=True)

        # Unzip into per-accession directories
        unzip_cmd = f"unzip -o {zip_path} -d {genome_dir}/batch_{i}_tmp"
        subprocess.run(unzip_cmd, shell=True, check=True)

        # Move each accession folder to genome_dir/<accession>
        tmp_dir = genome_dir / f"batch_{i}_tmp"
        ncbi_data = tmp_dir / "ncbi_dataset" / "data"
        if ncbi_data.exists():
            for acc_dir in ncbi_data.iterdir():
                dest = genome_dir / acc_dir.name
                if not dest.exists():
                    acc_dir.rename(dest)

        # Cleanup
        zip_path.unlink(missing_ok=True)
        subprocess.run(f"rm -rf {tmp_dir}", shell=True)

    print("Download complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtdb_dir",           default="data/gtdb")
    parser.add_argument("--out_dir",            default="data")
    parser.add_argument("--n_phyla",            type=int, default=20)
    parser.add_argument("--species_per_phylum", type=int, default=10)
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--datasets_bin",       default="datasets")
    parser.add_argument("--skip_download",      action="store_true",
                        help="Only write index, do not download FASTAs")
    args = parser.parse_args()

    gtdb_dir = Path(args.gtdb_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Parsing GTDB metadata ===")
    meta = parse_gtdb_metadata(gtdb_dir)
    print(f"Total GTDB representatives: {len(meta)}")

    print("\n=== Subsampling ===")
    sampled = stratified_subsample(meta, args.n_phyla, args.species_per_phylum, args.seed)

    index_path = out_dir / "species_index.tsv"
    sampled.to_csv(index_path, sep="\t", index=False)
    print(f"\nSpecies index written to {index_path}")

    acc_path = out_dir / "accessions_to_download.txt"
    sampled["accession"].to_csv(acc_path, index=False, header=False)
    print(f"Accession list written to {acc_path}")

    if not args.skip_download:
        print("\n=== Downloading FASTAs ===")
        download_genomes(sampled["accession"].tolist(), out_dir, args.datasets_bin)
    else:
        print("\n--skip_download set; skipping FASTA downloads.")


if __name__ == "__main__":
    main()
