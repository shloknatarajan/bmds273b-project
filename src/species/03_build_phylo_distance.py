"""
03_build_phylo_distance.py
--------------------------
Parses the GTDB Newick trees (bacterial + archaeal) and computes
pairwise phylogenetic distances between all representative species
in our subsampled index.

Why pre-compute?
  ete3 tree distance queries are O(depth); computing all pairs at
  probe time for every layer would be prohibitively slow. We cache
  the full symmetric distance matrix as a compressed NPZ file.

Outputs
-------
data/phylo/
    distance_matrix.npz  — {'distances': NxN float32, 'species': N-length str array}
    species_in_tree.txt  — species present in both the index and the tree

Usage
-----
  python 03_build_phylo_distance.py \
      --tree_dir data/trees \
      --species_index data/species_index.tsv \
      --out_dir data/phylo
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ete3 import Tree
except ImportError:
    raise ImportError("Install ete3: pip install ete3")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gtdb_tree(tree_path: Path) -> Tree:
    """
    GTDB Newick files use accession-based leaf names like:
      GB_GCA_000001405.26
    We need to match these to our species_index accessions.
    """
    print(f"  Loading tree from {tree_path} ...")
    t = Tree(str(tree_path), format=1)
    return t


def map_leaf_to_accession(leaf_name: str) -> str:
    """
    GTDB tree leaf names are prefixed with 'GB_' or 'RS_'.
    Strip prefix to get the plain GCA/GCF accession used in our index.
    """
    return re.sub(r"^(GB_|RS_)", "", leaf_name)


def build_distance_matrix(tree: Tree, accessions: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Computes pairwise patristic distances between all accessions
    present in the tree.

    Returns (distance_matrix, filtered_accessions).
    Accessions not found in the tree are dropped with a warning.
    """
    # Build a map from clean accession → leaf node
    leaf_map: dict[str, "TreeNode"] = {}
    for leaf in tree.iter_leaves():
        clean = map_leaf_to_accession(leaf.name)
        leaf_map[clean] = leaf

    # Filter to accessions actually in the tree
    found = [a for a in accessions if a in leaf_map]
    missing = [a for a in accessions if a not in leaf_map]
    if missing:
        print(f"  WARNING: {len(missing)} accessions not found in tree (dropped): {missing[:3]} ...")
    print(f"  Building {len(found)}×{len(found)} distance matrix ...")

    n = len(found)
    dmat = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            d = leaf_map[found[i]].get_distance(leaf_map[found[j]])
            dmat[i, j] = d
            dmat[j, i] = d

        if i % 20 == 0:
            print(f"    Progress: {i}/{n} rows done", end="\r")

    print()
    return dmat, found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_dir",      default="data/trees")
    parser.add_argument("--species_index", default="data/species_index.tsv")
    parser.add_argument("--out_dir",       default="data/phylo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_dir = Path(args.tree_dir)

    index = pd.read_csv(args.species_index, sep="\t")
    accessions = index["accession"].tolist()
    print(f"Total accessions in index: {len(accessions)}")

    all_dists: list[np.ndarray] = []
    all_accs:  list[str]        = []

    for tree_file in ["bac120.tree", "ar53.tree"]:
        fpath = tree_dir / tree_file
        if not fpath.exists():
            print(f"  {tree_file} not found, skipping.")
            continue
        tree = load_gtdb_tree(fpath)
        dmat, found_accs = build_distance_matrix(tree, accessions)
        all_dists.append(dmat)
        all_accs.extend(found_accs)

    if not all_accs:
        raise RuntimeError("No accessions matched any tree. Check tree filenames and accession formats.")

    # If we have both bac and arc subsets, combine into one block-diagonal matrix
    # (cross-domain distances are undefined; we set them to np.nan)
    if len(all_dists) == 2:
        n1, n2 = len(all_dists[0]), len(all_dists[1])
        full = np.full((n1 + n2, n1 + n2), np.nan, dtype=np.float32)
        full[:n1, :n1] = all_dists[0]
        full[n1:, n1:] = all_dists[1]
        final_dmat = full
    else:
        final_dmat = all_dists[0]

    out_path = out_dir / "distance_matrix.npz"
    np.savez_compressed(out_path, distances=final_dmat,
                        accessions=np.array(all_accs))
    print(f"\nSaved distance matrix ({final_dmat.shape}) to {out_path}")

    # Also write a plain list of accessions that have tree entries
    with open(out_dir / "accessions_in_tree.txt", "w") as f:
        for a in all_accs:
            f.write(a + "\n")
    print(f"Accessions present in tree: {len(all_accs)}")


if __name__ == "__main__":
    main()
