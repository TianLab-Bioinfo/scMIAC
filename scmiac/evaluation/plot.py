import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import numpy as np
import pandas as pd
import anndata



def plot_modality_gex(adata_cm, adata_rna_raw, adata_ga_raw, modality='RNA', gene='CD8A', basis='latent_umap', vmax=None):
    
    if modality == 'RNA':
        gene_expression_raw = adata_rna_raw[:, gene].X.toarray().flatten()
        gene_expression = pd.Series(gene_expression_raw, index=adata_rna_raw.obs_names)
    elif modality == 'ATAC':
        gene_expression_raw = adata_ga_raw[:, gene].X.toarray().flatten()
        gene_expression = pd.Series(gene_expression_raw, index=adata_ga_raw.obs_names)

    adata_cm.obs[f'{modality}_{gene}'] = np.nan

    matching_cells = adata_cm.obs.index.intersection(gene_expression.index)
    adata_cm.obs.loc[matching_cells, f'{modality}_{gene}'] = gene_expression.loc[matching_cells]

    sc.pl.embedding(
        adata_cm, 
        basis=basis, 
        color=[f'{modality}_{gene}'], 
        legend_loc='on data', 
        vmax=vmax
    )