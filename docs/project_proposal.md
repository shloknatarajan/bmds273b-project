# What Do DNA Language Models Learn? A Layer-wise and Phylogenetic Analysis of Genomic Representations

Mihir Borkar, Maya Drusinsky, Vivian Utti, Shlok Natarajan

## Research Problem

Currently, DNA language models are not very interpretable. How can we use embeddings from these models to assess their performance and what they are truly learning? We aim to explore two distinct areas:

1. Classifying regulatory role (predicting type of sequence, e.g. regulatory vs. gene)
2. Species prediction from sequence

## Datasets

For our first task, classifying the regulatory role of a sequence, we will utilize GENCODE (gene annotations) and ENCODE (regulatory element annotations). GENCODE and ENCODE are funded by the National Human Genome Research Institute (NHGRI), with the goals of annotating evidence-based gene features and functional elements in the human genome, respectively. We will use the most recent release, which contains over 78,000 human genes. Since regulatory regions can often intersect with gene bodies, we will label a “gene” as a segment intersecting a GENCODE gene body without an annotated cis-regulatory element (cCRE) on ENCODE and label "regulatory” as a segment intersecting a cCRE within an intron or between genes (intergenic).

For our second task, species prediction from sequence, we will use data from the Genome Taxonomy Database (GTDB), which provides a standard taxonomy for bacteria and archaea, based on genome phylogeny. We will use the most recent release (232), which spans 901,341 genomes from almost 200,000 species. This data will provide ground-truth species labels and evolutionary distances between species, whose actual genetic sequences we will pull from the National Center for Biotechnology Information (NCBI). For possible expansion to eukaryotic species, we will utilize the NCBI Reference Sequence Database for eukaryotic genomes.

## Methods

For the regulatory vs gene classification task, context length becomes a binding constraint as regulatory elements are largely defined by position relative to transcription sites and gene bodies, which can be hundreds of kb away. For this task, 3 models could be promising:

- **HyenaDNA:** Hyena long-convolution model that handles up to ~1M nucleotides at single-nucleotide resolution.
- **Caduceus:** a bidirectional Mamba (state-space) model with reverse-complement equivariance built into the architecture.
- **Evo 2 7B:** an autoregressive StripedHyena foundation model trained on ~9.3T nucleotides across ~128k genomes spanning spanning bacteria, archaea, and eukaryotes.

We plan to freeze each model’s weights and extract per-token embeddings from every layer on long genomic windows. From GENCODE gene bodies and ENCODE cCRE annotations, we will draw matched >10 kb windows (e.g., length- and GC-matched) and mean-pool token embeddings within each window to obtain a single embedding vector per region. On top of these frozen embeddings, we will train a small classification model (e.g., logistic regression or a shallow MLP) to discriminate “gene body” vs. “regulatory element.” By training identical probes on embeddings from each layer, we can obtain layer-wise performance profiles for each architecture, allowing us to localize where in the network regulatory vs. gene identity emerges. As baselines, we will train simple models directly on raw sequence features (e.g., k-mer TF–IDF or a shallow CNN on the same windows) to ensure that any performance gains are not solely driven by trivial composition differences.

For the species from sequence task, context length is much less important as short fragments carry enough signal to be signal; however, cross species pre-training is vital. For this task, we’ve selected 3 promising models:

- **Nucleotide Transformer v2 2.5B:** A multi-species BERT-style transformer from InstaDeep pretrained on genomes from 850+ species.
- **DNABERT-S:** A DNABERT-2 variant fine-tuned with a contrastive species-aware objective to produce embeddings that cluster by taxonomy.
- **Evo 2 7B:** An autoregressive StripedHyena foundation model trained on ~9.3T nucleotides across ~128k genomes spanning spanning bacteria, archaea, and eukaryotes (same as above).

We aim to evaluate in two ways:

1. **Binary prediction:** we draw random fragments from a diverse set of species, take the model's embedding for each fragment, and train a simple linear classifier to identify the species, reporting standard classification metrics of AUPRC, AUC, and F1 Score.
2. **Embedding distance:** we take pairs of fragments and ask whether the distance between their embeddings (cosine or Euclidean) tracks the true evolutionary distance between their species. The idea is that if a model has genuinely internalized biology, closely related species should sit close together in its representation space.

## Evaluation Criteria

For the binary classification tasks, we will report AUPRC, F1-score, and AUC. For the embedding distance between species task, we will compute pairwise cosine and Euclidean distances between fragment embeddings and measure their correlation with ground-truth phylogenetic distances from TDB using Spearman’s rank correlation. Both probes are run at every layer of each model rather than only the final layer, allowing us to compare early vs. late representations and localize where different types of biological information are encoded. To yield rigorous insights, we will also compare early vs. late layer representations to understand where information is encoded, rather than relying solely on final-layer embeddings.

Success would be to show that DNA language models embeddings capture biologically meaningful signal beyond simple sequence features, and that the signal can be localized across layers. Our two main goals are to demonstrate whether:

1. Simple probes on frozen embeddings outperform sequence-based baselines on both regulatory classification and species prediction.
2. Embedding distances reflect true biological relationships, such as evolutionary similarity between species.

Our feasible target is to demonstrate that learned embeddings outperform sequence-based baselines on both tasks and show non-trivial correlation (e.g., Spearman ρ > 0.3) with phylogenetic distance. Our ambitious target is to identify clear, architecture-dependent patterns in layer-wise information encoding (e.g., regulatory information emerging later in long-context models), and achieve strong phylogenetic alignment (e.g., ρ > 0.6), indicating that models capture meaningful evolutionary structure.

## Analysis or Application

For our analysis, we will probe how biological information is encoded across layers of each DNA language model by systematically evaluating embeddings on two complementary tasks: species prediction and regulatory vs. gene classification. By training simple, standardized classifiers on frozen embeddings from every layer, we can generate layer-wise performance curves that reveal where different types of biological signal emerge. We will analyze embedding geometry directly by measuring whether distances between sequence embeddings correlate with known evolutionary distances between species. This allows us to move beyond task performance and assess whether the models organize biological relationships in a meaningful latent space.

From these analyses, we hope to extract insights into both model behavior and biological representation. Specifically, we aim to determine whether different architectures (e.g., short-context transformers vs. long-context state-space models) encode distinct types of information at different depths, and whether long-range context meaningfully improves the representation of regulatory elements. We are also interested in whether species-level information emerges early (suggesting reliance on local sequence patterns) or later (indicating more abstract representations), and whether embedding spaces reflect true phylogenetic structure. This project seeks to answer whether these models are learning biologically relevant abstractions or merely exploiting statistical shortcuts in DNA sequences.

The potential impact of this work lies in improving both the interpretability and reliability of genomic foundation models in biomedicine. If we can identify where and how meaningful biological signals are encoded, it becomes easier to trust and deploy these models in downstream applications such as variant effect prediction and functional annotation of noncoding regions. Demonstrating that embedding spaces align with evolutionary or regulatory structure could enable new unsupervised tools for discovering novel functional elements or characterizing unknown species. This work will contribute to bridging the gap between high-performing DNA language models and their practical, interpretable use in genomics and precision medicine.

## References

1. Nguyen, E., Poli, M., Faizi, M., Thomas, A., Birch-Sykes, C., Wornow, M., Patel, A., Rabideau, C., Massaroli, S., Bengio, Y., Ermon, S., Baccus, S. A., & Ré, C. (2023). *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution.* arXiv:2306.15794. NeurIPS 2023.
2. Schiff, Y., Kao, C.-H., Gokaslan, A., Dao, T., Gu, A., & Kuleshov, V. (2024). *Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling.* arXiv:2403.03234. ICML 2024, PMLR 235.
3. Brixi, G., Durrant, M. G., Ku, J., Poli, M., et al., Hsu, P. D., & Hie, B. L. (2025). *Genome modeling and design across all domains of life with Evo 2.* bioRxiv 2025.02.18.638918. Nature, 2026.
4. Dalla-Torre, H., Gonzalez, L., Mendoza-Revilla, J., Lopez Carranza, N., Grzywaczewski, A. H., Oteri, F., Dallago, C., Trop, E., de Almeida, B. P., Sirelkhatim, H., Richard, G., Skwark, M., Beguir, K., Lopez, M., & Pierrot, T. (2024). *Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics.* Nature Methods, 22(2), 287–297. https://doi.org/10.1038/s41592-024-02523-z
5. Zhou, Z., Wu, W., Ho, H., Wang, J., Shi, L., Davuluri, R. V., Wang, Z., & Liu, H. (2024). *DNABERT-S: Pioneering Species Differentiation with Species-Aware DNA Embeddings.* arXiv:2402.08777. Bioinformatics, 2024.
