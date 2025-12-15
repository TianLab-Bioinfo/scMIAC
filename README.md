# scMIAC: Single-Cell Multi-modality Integration via cell type filtered Anchors using Contrastive learning

## Overview

scMIAC is a comprehensive framework for single-cell multi-modality data integration, designed to tackle the most challenging problem in single-cell integration: **diagonal integration** (integrating unpaired cells from different feature spaces across modalities).

The methodological innovations of scMIAC include:
- scMIAC utilizes cell type information to select high-quality anchor cells for contrastive learning, improving integration of challenging cells such as imbalanced, rare, or isolated cell types.
- scMIAC innovatively introduces contrastive learning to diagonal integration task, where previous methods could only be applied to horizontal or vertical integration scenarios.
- As a diagonal integration approach, scMIAC preserves each modality's original biological characteristics through modality-specific VAEs, which serves as a regularizer preventing over-emphasis on modality alignment.

![scMIAC framework](https://raw.githubusercontent.com/TianLab-Bioinfo/scMIAC/main/figs/fig1.webp)

---

## Installation

```bash
conda create -n scmiac python=3.11
conda activate scmiac
pip install scmiac
```

---

## Usage

scMIAC provides two usage modes:

### 1. CLI Mode - Quick start for non-interactive, command-line workflow:

```bash
scmiac train \
  --rna-h5ad data/10x/input/adata_rna_10x.h5ad \
  --atac-h5ad data/10x/input/adata_atac_10x.h5ad \
  --output-dir data/10x/output/scmiac_results/ \
  --rna-latent-key X_pca \
  --atac-latent-key lsi49 \
  --rna-celltype-key cell_type \
  --atac-celltype-key pred

scmiac train -h  # For viewing all available parameters
```

**MUST Required parameters:**
- `--rna-h5ad`: Path to RNA AnnData file
- `--atac-h5ad`: Path to ATAC AnnData file
- `--output-dir`: Output directory

**Output files:**
- `anchors.csv`: Anchor pairs
- `rna_vae.pth`: RNA VAE model weights
- `atac_vae.pth`: ATAC VAE model weights
- `rna_embeddings.csv`: RNA cell embeddings
- `atac_embeddings.csv`: ATAC cell embeddings
- `scmiac_latent_umap.png`: UMAP visualization


### 2. API Mode - For flexible research and experimentation:

Refer to the [Full API Tutorial & Examples](https://github.com/TianLab-Bioinfo/scMIAC/blob/main/docs/tutorial.ipynb) for detailed usage and examples.

---

## Reproducing Paper Results

The `scripts/` directory contains all experimental code for reproducing the results in the scMIAC paper, including:
- Competing method comparisons
- Ablation studies
- Hyperparameter experiments
- Evaluation framework

For detailed information, see [scripts/README.md](scripts/README.md).


