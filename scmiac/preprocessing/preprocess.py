import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.sparse import issparse
import sklearn
from sklearn.preprocessing import normalize
import torch




def preprocessing_atac(adata_atac, binary=True, n_top_peaks=None):
    """
    Preprocess single-cell ATAC-seq data.

    Parameters:
        adata (AnnData): AnnData object containing ATAC-seq data.
        binary (bool): Whether to binarize the expression matrix (set values greater than 1 to 1), default is True.
        n_top_peaks (int): Number of highly variable genes to retain, default is 30,000.

    Returns:
        AnnData: A new AnnData object containing only the specified number of highly variable genes.

    Example:
        adata_processed = preprocessing_atac(adata, binary=True, n_top_peaks=30000)
    """
    print('Raw dataset shape: {}'.format(adata_atac.shape))
    if not issparse(adata_atac.X):
        adata_atac.X = scipy.sparse.csr_matrix(adata_atac.X)
    if binary:
        adata_atac.X[adata_atac.X > 1] = 1
    if n_top_peaks is not None:
        # inplace=False: Returns a new AnnData object without modifying the original adata
        # subset=True: Retains only the specified number of highly variable genes
        adata_atac = sc.pp.highly_variable_genes(adata_atac, n_top_genes=n_top_peaks, subset=True, inplace=False)
        print('Processed dataset shape: {}'.format(adata_atac.shape))
    return adata_atac





def tfidf(X):
    """
    Compute the TF-IDF (Term Frequency-Inverse Document Frequency) matrix for feature normalization.

    Parameters:
        X (scipy.sparse.csr_matrix or np.ndarray): Feature matrix, typically a sparse matrix.

    Returns:
        scipy.sparse.csr_matrix or np.ndarray: TF-IDF normalized matrix, maintaining the input matrix format.

    Example:
        tfidf_matrix = tfidf(X)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf





def run_lsi(adata_atac, n_components=50,post_normalizing=True):
    """
    Perform TF-IDF normalization, standardization, and Latent Semantic Indexing (LSI) dimensionality reduction on single-cell ATAC-seq data.

    Parameters:
        adata_atac (AnnData): AnnData object containing ATAC-seq data.
        n_components (int): Number of components for LSI dimensionality reduction, default is 50.

    Returns:
        np.ndarray: Feature matrix after LSI dimensionality reduction.

    Example:
        lsi_result = perform_lsi(adata_atac, n_components=50)
    """
    print('tfidf')
    X_tfidf = tfidf(adata_atac.X)

    print('normalizing')
    X_norm = normalize(X_tfidf, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)

    print('svd')
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components=n_components)[0]

    # X_lsi = X_lsi[:, 1:]
    if post_normalizing:
        print('post normalizing')
        X_lsi -= X_lsi.mean(axis=1, keepdims=True)
        X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)

    adata_atac.obsm['lsi']=X_lsi
    return adata_atac





def run_umap(adata, use_rep='lsi'):
    """
    Compute a neighbor graph and generate UMAP embeddings based on the specified representation (e.g., LSI).

    Parameters:
        adata (AnnData): AnnData object containing single-cell data.
        use_rep (str): Representation method used for neighbor graph and UMAP embedding, e.g., 'lsi'.

    Returns:
        adata: Updated AnnData object with UMAP results added under the key use_rep + "_umap".

    Example:
        run_umap(adata_atac, use_rep='lsi')
    """

    sc.pp.neighbors(adata, use_rep=use_rep)
    
    sc.tl.umap(adata)
    
    umap_key = use_rep + "_umap"
    adata.obsm[umap_key] = adata.obsm["X_umap"]
    
    del adata.obsm["X_umap"]
    print(f"UMAP embedding is stored in adata.obsm['{umap_key}']")

    return adata





def nmf_pytorch(X, n_components, n_iter=2000, tol=1e-6, device='cuda:0'):
    """
    Implement Non-negative Matrix Factorization (NMF) using PyTorch and run it on a specified device.

    Parameters:
        X (np.ndarray or torch.Tensor): The matrix to be factorized (shape: m x n). If np.ndarray, it will be converted to a tensor.
        n_components (int): Number of components after NMF, i.e., the number of columns in W and rows in H.
        n_iter (int): Maximum number of iterations, default is 2000.
        tol (float): A small constant to prevent division by zero, default is 1e-6.
        device (str): The computing device, e.g., 'cuda:0' or 'cpu'.

    Returns:
        W (torch.Tensor): Factorized W matrix (shape: m x n_components).
        H (torch.Tensor): Factorized H matrix (shape: n_components x n).

    Notes:
        The optimization follows the multiplicative update rules:
        - H update: H *= (W.T @ X) / (W.T @ W @ H + tol)
        - W update: W *= (X @ H.T) / (W @ H @ H.T + tol)

        Every 100 iterations, the loss (Frobenius norm squared of the reconstruction error) is computed and printed to monitor convergence.

    Example:
        W, H = nmf_pytorch(X, n_components=50, n_iter=2000, tol=1e-6, device='cuda:3')
    """

    X = torch.tensor(X, dtype=torch.float32).to(device)
    
    m, n = X.shape
    W = torch.rand(m, n_components, device=device)
    H = torch.rand(n_components, n, device=device)
    
    for i in range(n_iter):
        H *= (W.T @ X) / (W.T @ W @ H + tol)
        W *= (X @ H.T) / (W @ H @ H.T + tol)
        
        if (i + 1) % 100 == 0:
            X_hat = W @ H
            loss = torch.norm(X - X_hat, p='fro') ** 2
            print(f"Iteration {i + 1}/{n_iter}, Loss: {loss.item()}")
    
    return W, H



def run_nmf(adata_atac, n_components=50, n_iter=10000, device='cuda:0'):
    """
    Perform TF-IDF normalization, standardization, and Non-negative Matrix Factorization (NMF) dimensionality reduction on single-cell ATAC-seq data.

    Parameters:
        adata_atac (AnnData): AnnData object containing ATAC-seq data.
        n_components (int): Number of components for NMF dimensionality reduction, default is 50.
        n_iter (int): Maximum number of iterations for the NMF algorithm, default is 10,000.
        device (str): Computing device to use, e.g., 'cuda:0'.

    Returns:
        np.ndarray: Feature matrix after NMF dimensionality reduction.

    Example:
        X_nmf = run_nmf(adata_atac, n_components=50, n_iter=10000, device='cuda:3')
    """

    print('tfidf')
    X_tfidf = tfidf(adata_atac.X).toarray()

    print('normalizing')
    X_norm = normalize(X_tfidf, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_cpu = torch.tensor(X_norm)
    
    print('nmf')
    W_gpu, H_gpu = nmf_pytorch(X_cpu, n_components=n_components, n_iter=n_iter, device=device)
    X_nmf = W_gpu.cpu().numpy()

    print('post normalizing')
    X_nmf -= X_nmf.mean(axis=1, keepdims=True)
    X_nmf /= X_nmf.std(axis=1, ddof=1, keepdims=True)

    return X_nmf
