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

Inputs
------
processed_data/
    species_index.tsv   — accession column uses GTDB-style prefixes
                          (e.g. ``GB_GCA_964402295.1``, ``RS_GCF_...``)
processed_data/trees/
    bac120_r232.tree  (or bac120.tree)
    ar53_r232.tree    (or ar53.tree)

Outputs
-------
processed_data/phylo/
    distance_matrix.npz  — {'distances': NxN float32, 'accessions': N-length str array}
    accessions_in_tree.txt  — accessions present in both the index and a tree

Usage
-----
  python 03_build_phylo_distance.py \
      --tree_dir data/trees \
      --species_index processed_data/species_index.tsv \
      --out_dir processed_data/phylo
"""

import argparse
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
    GTDB Newick leaf names already include the source prefix
    (``GB_GCA_...`` or ``RS_GCF_...``), matching the accession column
    in our species_index.tsv. We therefore keep the leaf name as-is.
    """
    print(f"  Loading tree from {tree_path} ...")
    t = Tree(str(tree_path), format=1, quoted_node_names=True)
    return t


def find_tree_file(tree_dir: Path, prefix: str) -> Path | None:
    """
    Locate a GTDB tree file. GTDB ships files like ``bac120_r232.tree``
    or ``ar53_r232.tree``; older docs reference ``bac120.tree``.
    Accept either naming convention.
    """
    candidates = [
        tree_dir / f"{prefix}.tree",
        tree_dir / f"{prefix}_r232.tree",
    ]
    candidates += sorted(tree_dir.glob(f"{prefix}*.tree"))
    for c in candidates:
        if c.exists():
            return c
    return None


def build_distance_matrix(tree: Tree, accessions: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Computes pairwise patristic distances between all accessions
    present in the tree.

    Returns (distance_matrix, filtered_accessions).
    Accessions not found in the tree are dropped with a warning.
    """
    leaf_map: dict[str, "TreeNode"] = {}
    for leaf in tree.iter_leaves():
        leaf_map[leaf.name] = leaf

    found = [a for a in accessions if a in leaf_map]
    missing = [a for a in accessions if a not in leaf_map]
    if missing:
        print(f"  WARNING: {len(missing)} accessions not found in tree (dropped). "
              f"First few: {missing[:3]}")
    print(f"  Building {len(found)}x{len(found)} distance matrix ...")

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
    parser.add_argument("--tree_dir",      default="processed_data/trees")
    parser.add_argument("--species_index", default="processed_data/species_index.tsv")
    parser.add_argument("--out_dir",       default="processed_data/phylo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_dir = Path(args.tree_dir)

    index = pd.read_csv(args.species_index, sep="\t")
    accessions = index["accession"].tolist()
    print(f"Total accessions in index: {len(accessions)}")

    bac_accs = set(index.loc[index["domain"] == "Bacteria", "accession"])
    arc_accs = set(index.loc[index["domain"] == "Archaea",  "accession"])
    print(f"  Bacteria: {len(bac_accs)}, Archaea: {len(arc_accs)}")

    all_dists: list[np.ndarray] = []
    all_accs:  list[str]        = []

    for tree_prefix, expected_accs in [("bac120", bac_accs), ("ar53", arc_accs)]:
        if not expected_accs:
            print(f"  No accessions in {tree_prefix} domain, skipping.")
            continue
        fpath = find_tree_file(tree_dir, tree_prefix)
        if fpath is None:
            print(f"  {tree_prefix}*.tree not found in {tree_dir}, skipping.")
            continue
        tree = load_gtdb_tree(fpath)
        dmat, found_accs = build_distance_matrix(tree, sorted(expected_accs))
        all_dists.append(dmat)
        all_accs.extend(found_accs)

    if not all_accs:
        raise RuntimeError(
            "No accessions matched any tree. Check that --tree_dir points "
            "to the GTDB Newick files and that they have been gunzipped."
        )

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

    with open(out_dir / "accessions_in_tree.txt", "w") as f:
        for a in all_accs:
            f.write(a + "\n")
    print(f"Accessions present in tree: {len(all_accs)}")


if __name__ == "__main__":
    main()
