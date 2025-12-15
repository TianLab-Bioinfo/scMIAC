# scMIAC Package Structure

This directory contains the implementation of the scMIAC package, organized into preprocessing, modeling, and evaluation modules.

## Directory Structure

```
scmiac/
├── preprocessing/              # Data preprocessing and anchor generation
│   ├── anchors.py              # CCA and mutual nearest neighbor anchor search
│   ├── preprocess.py           # ATAC preprocessing, LSI, NMF, UMAP
│   ├── cosg.py                 # COSG feature selection
│   └── atacannopy.py           # ATAC cell type annotation
│
├── modeling/                   # Models, datasets, and loss functions
│   ├── dataset.py              # PyTorch Dataset definitions
│   ├── model.py                # VAE encoder and decoder
│   ├── loss.py                 # Loss functions (VAE, contrastive learning)
│   └── scmiac.py               # Core API (anchor finding, training, inference)
│
├── evaluation/                 # Evaluation and visualization
│   ├── benchmark.py            # Integration quality metrics
│   └── plot.py                 # Visualization utilities
│
├── utils.py                    # Utility functions
├── cli.py                      # Command-line interface
└── __init__.py                 # Package entry point
```

## Core API

The scMIAC workflow consists of four main functions:

- **find_anchors**: Find anchor cell pairs between RNA and ATAC modalities
- **preprocess**: Initialize VAE models and create data loaders
- **train_model**: Train models with contrastive learning
- **model_inference**: Generate integrated embeddings

```python
from scmiac import find_anchors, preprocess, train_model, model_inference

# Find anchor pairs
anchor_df = find_anchors(adata_rna, adata_atac, ct_filter=True)

# Preprocess and initialize models
rna_vae, atac_vae, all_cells_loader, anchor_cells_loader = preprocess(
    adata_rna, adata_atac, anchor_df, latent_dim=30
)

# Train models
rna_vae, atac_vae = train_model(
    rna_vae, atac_vae, all_cells_loader, anchor_cells_loader,
    num_epoches=2000, lambda_contra=300
)

# Generate embeddings
rna_embeddings, atac_embeddings = model_inference(
    rna_vae, atac_vae, all_cells_loader
)
```

