Split: 140 train / 20 val / 40 test species. Species level split to avoid data leakage.
10000 fragments from 200 genomes
Split sizes (fragments): train: 7000, val: 1000, test: 2000

fragments.tsv contains one row per species. Columns are:
  frag_id: Unique ID for this fragment, built from accession + contig + position (ex: GCA_964402295.1_contig1_0)
  accession: The NCBI genome accession it came from (ex: GCA_964402295.1)
  species: GTDB species name (ex: s__Escherichia coli)
  phylum: GTDB phylum name (ex: p__Proteobacteria)
  domain: Bacteria or Archaea (ex: Bacteria)
  split: Whether this fragment is in train/val/test (ex: train)
  contig: Which contig within the genome it came from (ex: JAKTOP010000001.1)
  start: Start position in the contig (bp) (ex: 0)
  end: End position in the contig (bp) (ex: 1024)
  seq: The actual DNA sequence, 1024 characters of A/T/G/C (ex: ATGCTTGA...)

species_split.json contains which split each species belongs in.
