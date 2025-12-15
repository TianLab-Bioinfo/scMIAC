import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import ranksums
from scipy.stats import spearmanr, pearsonr, kendalltau
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from .cosg import cosg

def filter_to_common_genes(adata1, adata2):
    """
    Retain only the common genes between two AnnData objects and update their raw data accordingly.

    Parameters:
        adata1 (AnnData): The first AnnData object, must contain raw data.
        adata2 (AnnData): The second AnnData object, must contain raw data.

    Returns:
        AnnData: The modified adata1 and adata2 objects with raw data filtered to the common genes.

    Note:
        This function does not modify the main data matrix (adata.X), only the raw data.

    Example:
        adata1, adata2 = filter_to_common_genes(adata1, adata2)
    """

    if adata1.raw is None or adata2.raw is None:
        raise ValueError("Both AnnData objects must have raw data.")

    common_genes = adata1.raw.var_names.intersection(adata2.raw.var_names)
    
    adata1.raw = adata1.raw.to_adata().copy()[:, list(common_genes)]
    adata2.raw = adata2.raw.to_adata().copy()[:, list(common_genes)]

    adata1 = adata1[:, adata1.var_names.isin(common_genes)]
    adata2 = adata2[:, adata2.var_names.isin(common_genes)]

    return adata1, adata2



def get_pseudo_bulk_mtx(sc_mtx, labels, genes, mode="mean"):
    """
    Generate pseudo-bulk expression matrix by aggregating single-cell expression profiles based on given cell-type labels.

    Parameters:
        sc_mtx (array-like or sparse matrix): Single-cell gene expression matrix with shape (cells, genes).
        labels (list or array-like): Cell-type or group labels for each cell (length should match the number of cells).
        genes (list): List of gene names corresponding to the columns in sc_mtx.
        mode (str, optional): Aggregation method to create pseudo-bulk expression. Options:
            - "mean": Computes the mean expression per gene for each cell group (default).
            - "sum": Computes the total expression per gene for each cell group.

    Returns:
        pd.DataFrame: Pseudo-bulk expression matrix (groups × genes), with groups as index and gene names as columns.

    Raises:
        ValueError: If an unsupported aggregation mode is provided.

    Example:
        pseudo_bulk_mtx = get_pseudo_bulk_mtx(sc_mtx, labels, genes, mode="mean")
    """

    expression_df = pd.DataFrame(sc_mtx, index=labels, columns=genes)

    expression_df['group'] = labels
    if mode == "mean":
        out_mtx = expression_df.groupby('group').mean()
    elif mode == "sum":
        out_mtx = expression_df.groupby('group').sum()
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'mean' or 'sum'.")
    return out_mtx



def get_kendall_genes(adata_ga, global_marker_df, neighbor_marker_dict,each_cluster_genes=100):
    """
    Identify genes for calculating Kendall’s tau correlations by integrating global marker genes, neighborhood marker genes, and COSG-derived markers from clustering results.

    Parameters:
        adata_ga (AnnData): AnnData object containing gene activity (GA) data for scATAC-seq cells.
        global_marker_df (pd.DataFrame): DataFrame of global marker genes across cell types or clusters.
        neighbor_marker_dict (dict): Dictionary containing neighborhood marker genes structured by cell types or clusters.
        each_cluster_genes (int, optional): Number of top COSG genes to select per cluster. Default is 100.

    Returns:
        np.ndarray: Array containing the unified set of marker genes for Kendall’s tau computation.

    Example:
        union_genes = get_kendall_genes(adata_ga, global_marker_df, neighbor_marker_dict, each_cluster_genes=100)
    """
    print('clustering ga')
    sc.pp.neighbors(adata_ga, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata_ga)
    
    # cosg
    cosg(adata_ga,
         key_added='cosg',
         mu=100,
         use_raw=True,
         expressed_pct=0.1,
         remove_lowly_expressed=True,
         n_genes_user=each_cluster_genes,
         groupby='leiden'
    )
    
    ga_global_marker_df = pd.DataFrame(adata_ga.uns['cosg']['names'])
    union_gm_ga = np.unique(ga_global_marker_df.values.ravel())
    
    union_gm = np.unique(global_marker_df.values.ravel())
    
    union_nm = set()
    for sub_dict in neighbor_marker_dict.values():
        for value in sub_dict.values():
            union_nm.update(value)
    union_nm = np.array(list(union_nm))
    
    union_all = np.union1d(union_gm_ga, np.union1d(union_gm, union_nm))
    
    print("Union length:", len(union_all))
    
    return union_all



def get_cor_mtx(query_exp, pb_ref, cor_method='kendall', genes=None):
    """
    Calculate the similarity matrix between query expression data and pseudo-bulk reference profiles using correlation coefficients.

    Parameters:
        query_exp (pd.DataFrame): Expression data for query cells (rows: cells, columns: genes).
        pb_ref (pd.DataFrame): Pseudo-bulk reference profiles (rows: reference types, columns: genes).
        cor_method (str, optional): Correlation method ('kendall', 'spearman', 'pearson'). Default is 'kendall'.
        genes (list, optional): List of genes to subset data before calculating correlations.

    Returns:
        pd.DataFrame: Similarity matrix between query cells and reference types.

    Example:
        cor_matrix = get_cor_mtx(query_exp, pb_ref, cor_method='kendall', genes=my_genes_list)
    """

    if cor_method == 'kendall':
        cor_func = kendalltau
    elif cor_method == 'spearman':
        cor_func = spearmanr
    elif cor_method == 'pearson':
        cor_func = pearsonr
    else:
        raise ValueError("Invalid correlation function. Choose 'kendall', 'spearman', or 'pearson'.")

    if genes is not None:
        query_exp = query_exp.loc[:, genes]
        pb_ref = pb_ref.loc[:, genes]

    num_query_cells = query_exp.shape[0]
    num_ref_types = pb_ref.shape[0]
    cor_mtx = np.zeros((num_query_cells, num_ref_types))

    for i in range(num_query_cells):
        for j in range(num_ref_types):
            query_cell = query_exp.iloc[i, :].values
            ref_type = pb_ref.iloc[j, :].values
            cor_value, _ = cor_func(query_cell, ref_type)
            cor_mtx[i, j] = cor_value
        if i % 2000 == 0:
            print(f"Processed {i} query cells")

    cor_mtx_df = pd.DataFrame(cor_mtx, columns=pb_ref.index, index=query_exp.index)
    return cor_mtx_df



def get_kendall_pred(cor_mtx_df):
    """
    Predict cell types based on Kendall's correlation similarity matrix by identifying the reference type with the highest correlation for each cell.

    Parameters:
        cor_mtx_df (pd.DataFrame): Correlation matrix (rows: query cells, columns: reference types).

    Returns:
        pd.DataFrame: A DataFrame containing:
            - kendall_pred: Predicted reference type with the highest correlation.
            - max_cor: Highest correlation value for each cell.
            - diff: Difference between the highest and second-highest correlation values, representing prediction confidence.

    Example:
        pred_df = get_kendall_pred(cor_mtx_df)
    """

    kendall_pred = cor_mtx_df.idxmax(axis=1)
    max_cor = cor_mtx_df.max(axis=1)
    second_max_cor = cor_mtx_df.apply(lambda x: x.nlargest(2).iloc[-1], axis=1)
    diff = max_cor - second_max_cor
    result_df = pd.DataFrame({
        'kendall_pred': kendall_pred,
        'max_cor': max_cor,
        'diff': diff
    })
    return result_df


def get_global_marker_df(adata, groupby='cell_type', each_ct_gene_num= 100,use_raw=True):
    """
    Identify global marker genes for each cell type or cluster using the COSG algorithm.

    Parameters:
        adata (AnnData): AnnData object containing single-cell expression data.
        groupby (str, optional): Key in adata.obs specifying cell type or cluster labels. Default is 'cell_type'.
        each_ct_gene_num (int, optional): Number of top marker genes to select for each cell type or cluster. Default is 100.
        use_raw (bool, optional): Whether to use raw data for identifying marker genes. Default is True.

    Returns:
        pd.DataFrame: DataFrame containing global marker genes for each cell type or cluster.

    Example:
        global_marker_df = get_global_marker_df(adata, groupby='cell_type', each_ct_gene_num=100)
    """

    cosg(adata,
         key_added='cosg',
         use_raw=use_raw, 
         mu=100,
         expressed_pct=0.1,
         remove_lowly_expressed=True,
         n_genes_user=each_ct_gene_num,
         groupby=groupby)

    global_marker_df = pd.DataFrame(adata.uns['cosg']['names'])
    return global_marker_df


def test_global_markers(cell_meta, sc_mtx_df, global_marker_df):
    """
    Evaluate cell-type predictions by performing a statistical test (Wilcoxon rank-sum) comparing the expression of predicted global marker genes against non-marker genes for each cell.

    Parameters:
        cell_meta (pd.DataFrame): Metadata DataFrame containing at least a column 'kendall_pred' indicating predicted cell types.
        sc_mtx_df (pd.DataFrame): Single-cell gene expression DataFrame (rows: cells, columns: genes).
        global_marker_df (pd.DataFrame): DataFrame of global marker genes per cell type (columns represent cell types).

    Returns:
        pd.DataFrame: Updated cell_meta DataFrame with an additional column:
            - GMSS: Global Marker Significance Score (-log10 of one-sided Wilcoxon rank-sum test p-value) representing confidence in cell-type predictions.

    Example:
        updated_cell_meta = test_global_markers(cell_meta, sc_mtx_df, global_marker_df)
    """

    gmss_values = []

    for idx, (i, row) in enumerate(cell_meta.iterrows()):
        cell_type = row['kendall_pred']
        if pd.isna(cell_type):
            gmss_value = 0
        else:
            marker_genes = global_marker_df[cell_type].dropna().values
            cell_expression = sc_mtx_df.loc[i]
            marker_gene_expression = cell_expression.loc[marker_genes]
            other_gene_expression = cell_expression.drop(marker_genes)
            stat, p_value = ranksums(marker_gene_expression, other_gene_expression)
            if stat > 0:
                p_value_one_sided = p_value / 2
            else:
                p_value_one_sided = 1 - p_value / 2
            gmss_value = -np.log10(p_value_one_sided)
        gmss_values.append(gmss_value)
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} cells")
    cell_meta['GMSS'] = gmss_values

    return cell_meta



def get_ct_similarity(pb_ref, global_marker_df,plot=False):
    """
    Calculate cell-type similarity based on Kendall’s tau correlation between pseudo-bulk reference profiles using global marker genes.

    Parameters:
        pb_ref (pd.DataFrame): Pseudo-bulk reference expression matrix (rows: cell types, columns: genes).
        global_marker_df (pd.DataFrame): DataFrame of global marker genes identified across cell types.
        plot (bool, optional): Whether to visualize the similarity matrix using a heatmap. Default is False.

    Returns:
        pd.DataFrame: Cell-type similarity matrix based on Kendall's tau correlation.

    Example:
        ct_similarity_df = get_ct_similarity(pb_ref, global_marker_df, plot=True)
    """
    
    union_gm = np.unique(global_marker_df.values.ravel())
    pb_ref = pb_ref.loc[:,union_gm]
    num_ref_types = pb_ref.shape[0]

    kendall_mtx = np.zeros((num_ref_types, num_ref_types))

    for i in range(num_ref_types):
        for j in range(i, num_ref_types): 
            if i == j:
                kendall_mtx[i, j] = 1 
            else:
                tau, _ = kendalltau(pb_ref.iloc[i, :], pb_ref.iloc[j, :])
                kendall_mtx[i, j] = tau
                kendall_mtx[j, i] = tau

    ct_similarity_df = pd.DataFrame(
        kendall_mtx, index=pb_ref.index, columns=pb_ref.index)

    if plot:
        kendall_mtx_df_percent = ct_similarity_df * 100
        g = sns.clustermap(kendall_mtx_df_percent, annot=True, fmt=".0f", cmap="coolwarm", cbar=True, vmin=0, vmax=100,
                           row_cluster=False, col_cluster=False, figsize=(6, 6))  # 设置图形大小
        plt.show()

    return ct_similarity_df


def get_neighbor_ct_dict(ct_similarity_df, threshold=0.6):
    """
    Generate a dictionary mapping each cell type to its neighboring cell types based on a similarity threshold.

    Parameters:
        ct_similarity_df (pd.DataFrame): Cell-type similarity matrix (rows and columns represent cell types).
        threshold (float, optional): Similarity threshold to define neighbors. Default is 0.6.

    Returns:
        dict: Dictionary with each cell type as keys and lists of neighboring cell types (similarity > threshold) as values.

    Example:
        neighbor_ct_dict = get_neighbor_ct_dict(ct_similarity_df, threshold=0.6)
    """

    neighbor_ct_dict = {}
    for cell_type in ct_similarity_df.index:
        high_similarity_cells = ct_similarity_df[cell_type][ct_similarity_df[cell_type] > threshold].index.tolist(
        )
        high_similarity_cells = [ct for ct in high_similarity_cells]
        neighbor_ct_dict[cell_type] = high_similarity_cells
    return neighbor_ct_dict


def get_neighbor_marker_ct(ct, neighbor_ct_dict, adata, groupby='cell_type', each_ct_gene_num=100,use_raw = True):
    """
    Identify marker genes that specifically distinguish a given cell type from its neighboring cell types using the COSG algorithm.

    Parameters:
        ct (str): Target cell type for identifying specific marker genes.
        neighbor_ct_dict (dict): Dictionary mapping cell types to their neighboring cell types.
        adata (AnnData): AnnData object containing single-cell expression data.
        groupby (str, optional): Key in adata.obs specifying cell type or cluster labels. Default is 'cell_type'.
        each_ct_gene_num (int, optional): Number of top marker genes selected per cell type. Default is 100.
        use_raw (bool, optional): Whether to use raw data for marker gene identification. Default is True.

    Returns:
        dict: Dictionary containing:
            - 'ct_markers': Array of marker genes specific to the target cell type.
            - 'bg_genes': Array of marker genes from neighboring cell types (background).

    Example:
        neighbor_marker_dict = get_neighbor_marker_ct('B_cell', neighbor_ct_dict, adata, each_ct_gene_num=100)
    """

    cosg(adata,
         key_added='cosg',
         use_raw=use_raw, 
         mu=100,
         expressed_pct=0.1,
         remove_lowly_expressed=True,
         n_genes_user=each_ct_gene_num,
         groupby=groupby,
         groups=neighbor_ct_dict[ct])
    ct_marker_df = pd.DataFrame(adata.uns['cosg']['names'])
    ct_markers = np.array(ct_marker_df[ct])
    bg_genes = np.unique(ct_marker_df.drop(ct, axis=1).values.flatten())
    ct_neighbor_marker_dict = dict(ct_markers=ct_markers, bg_genes=bg_genes)
    return ct_neighbor_marker_dict


def get_neighbor_marker_dict(adata, neighbor_ct_dict, global_marker_df, groupby='cell_type', each_ct_gene_num=100,use_raw = True):
    """
    Construct a comprehensive dictionary of marker genes distinguishing each cell type from its neighbors, utilizing COSG-derived neighbor-specific markers.

    Parameters:
        adata (AnnData): AnnData object containing single-cell expression data.
        neighbor_ct_dict (dict): Dictionary mapping cell types to their neighboring cell types.
        global_marker_df (pd.DataFrame): DataFrame of global marker genes for all cell types.
        groupby (str, optional): Key in adata.obs specifying cell type or cluster labels. Default is 'cell_type'.
        each_ct_gene_num (int, optional): Number of marker genes selected per cell type. Default is 100.
        use_raw (bool, optional): Whether to use raw data for marker identification. Default is True.

    Returns:
        dict: Nested dictionary mapping each cell type to its specific marker genes and background genes from neighbors.

    Example:
        neighbor_marker_dict = get_neighbor_marker_dict(adata, neighbor_ct_dict, global_marker_df, each_ct_gene_num=100)
    """

    neighbor_marker_dict = dict()
    for ct in neighbor_ct_dict.keys():
        print(ct)
        if len(neighbor_ct_dict[ct]) == 1:
            ct_markers_dict = dict(ct_markers=np.array(global_marker_df[ct]),
                                   bg_genes=np.unique(global_marker_df.drop(ct, axis=1).values.flatten()))
        else:
            ct_markers_dict = get_neighbor_marker_ct(
                ct, neighbor_ct_dict, adata, groupby=groupby, each_ct_gene_num=each_ct_gene_num,use_raw = use_raw)
        neighbor_marker_dict[ct] = ct_markers_dict
    return neighbor_marker_dict





def test_neighbor_markers(cell_meta, sc_mtx_df, neighbor_marker_dict):
    """
    Evaluate cell-type predictions using neighbor-specific marker genes by calculating a Neighbor Marker Significance Score (NMSS) through a Wilcoxon rank-sum test.

    Parameters:
        cell_meta (pd.DataFrame): Metadata with predicted cell types (column: 'kendall_pred').
        sc_mtx_df (pd.DataFrame): Single-cell expression matrix (rows: cells, columns: genes).
        neighbor_marker_dict (dict): Dictionary of neighbor-specific marker genes.

    Returns:
        pd.DataFrame: Updated cell_meta including 'NMSS' indicating significance scores for neighbor-specific marker genes.
    """
    nmss_values = []

    for idx, (i, row) in enumerate(cell_meta.iterrows()):
        cell_type = row['kendall_pred']

        if pd.isna(cell_type):
            nmss_value = 0
        else:
            marker_genes = neighbor_marker_dict[cell_type]['ct_markers']
            bg_genes = neighbor_marker_dict[cell_type]['bg_genes']

            cell_expression = sc_mtx_df.loc[i]

            marker_gene_expression = cell_expression.loc[marker_genes]
            other_gene_expression = cell_expression.loc[bg_genes]

            stat, p_value = ranksums(marker_gene_expression, other_gene_expression)
            if stat > 0:
                p_value_one_sided = p_value / 2
            else:
                p_value_one_sided = 1 - p_value / 2

            nmss_value = -np.log10(p_value_one_sided)

        nmss_values.append(nmss_value)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} cells")

    cell_meta['NMSS'] = nmss_values

    return cell_meta



def plot_density_with_annotation(cell_meta, ct, x):
    """
    Generate density plots for a specific metric ('GMSS', 'NMSS', etc.) for a given cell type, annotated with counts of true and false predictions.

    Parameters:
        cell_meta (pd.DataFrame): Metadata including the prediction metric and a boolean column 'kendall_pred_booltrue'.
        ct (str): Target cell type to plot.
        x (str): Column name of the metric to visualize.

    Returns:
        None (plots density visualization).
    """

    ct_cell_meta = cell_meta[cell_meta['kendall_pred'] == ct]

    total_count = len(ct_cell_meta)
    true_count = ct_cell_meta['kendall_pred_booltrue'].sum()
    false_count = total_count - true_count

    plt.figure(figsize=(5, 3))
    sns.kdeplot(data=ct_cell_meta, x=x, hue='kendall_pred_booltrue',
                fill=True, common_norm=True)
    plt.title(f'{x} Density Plot for {ct} Cells')

    annotation_text = f'Total: {total_count} cells\nTrue: {true_count} cells\nFalse: {false_count} cells'
    plt.text(0.7, 0.5, annotation_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top')

    plt.xlabel(x)
    plt.ylabel('Density')
    plt.legend(title='kendall_pred_booltrue', labels=['True', 'False'])
    plt.show()

    
def get_seed_cells(cell_meta, neighbor_ct_dict, quantile_global=0.5, quantile_neighbor=0.1):
    """
    Select seed cells based on thresholds calculated from global and neighbor-specific marker significance scores (GMSS, NMSS) and prediction confidence differences.

    Parameters:
        cell_meta (pd.DataFrame): Cell metadata containing 'kendall_pred', 'GMSS', 'NMSS', and 'diff'.
        neighbor_ct_dict (dict): Dictionary mapping each cell type to its neighboring types.
        quantile_global (float): Threshold quantile for cell types with a single neighbor.
        quantile_neighbor (float): Threshold quantile for cell types with multiple neighbors.

    Returns:
        pd.DataFrame: Updated cell_meta with a new binary column 'is_seed' indicating seed cells.
    """
    cell_meta['is_seed'] = 0
    cts = np.unique(cell_meta['kendall_pred'])
    total_seed_cells = 0

    for ct in cts:
        if len(neighbor_ct_dict[ct]) == 1:
            top_quantile = quantile_global
        else:
            top_quantile = quantile_neighbor
        ct_cells = cell_meta[cell_meta['kendall_pred'] == ct]
        gmss_threshold = ct_cells['GMSS'].quantile(1 - 0.05)
        nmss_threshold = ct_cells['NMSS'].quantile(1 - top_quantile)
        diff_threshold = ct_cells['diff'].quantile(1 - top_quantile)
        seed_cells = ct_cells[
            (ct_cells['GMSS'] >= gmss_threshold) |
            (ct_cells['NMSS'] >= nmss_threshold) |
            (ct_cells['diff'] >= diff_threshold)
        ]
        cell_meta.loc[seed_cells.index, 'is_seed'] = 1
        num_seed_cells = len(seed_cells)
        print(f"{ct}: {num_seed_cells}")
        total_seed_cells += num_seed_cells
    total_cells = len(cell_meta)
    seed_cell_percentage = (total_seed_cells / total_cells) * 100
    print(f"All: {total_seed_cells}, {seed_cell_percentage:.2f}%")

    return cell_meta




def get_seed_cells_topk(cell_meta, top_k=10):
    """
    Select top-K seed cells per cell type based on NMSS and prediction confidence difference.

    Parameters:
        cell_meta (pd.DataFrame): Metadata including 'kendall_pred', 'NMSS', and 'diff'.
        top_k (int): Number of top-ranked cells to select for each metric.

    Returns:
        pd.DataFrame: Updated cell_meta including binary 'is_seed' indicating seed cells.
    """
    cell_meta['is_seed'] = 0
    cts = np.unique(cell_meta['kendall_pred'])
    total_seed_cells = 0
    for ct in cts:
        ct_cells = cell_meta[cell_meta['kendall_pred'] == ct]
        if len(ct_cells) <= top_k:
            seed_cells = ct_cells
        else:
            top_nmss = ct_cells.nlargest(top_k, 'NMSS')
            top_diff = ct_cells.nlargest(top_k, 'diff')
            seed_cells_idx = top_nmss.index.union(top_diff.index)
            seed_cells = ct_cells.loc[seed_cells_idx]

        cell_meta.loc[seed_cells.index, 'is_seed'] = 1
        num_seed_cells = len(seed_cells)
        print(f"{ct}: {num_seed_cells}")
        total_seed_cells += num_seed_cells

    total_cells = len(cell_meta)
    seed_cell_percentage = (total_seed_cells / total_cells) * 100
    print(f"All: {total_seed_cells}, {seed_cell_percentage:.2f}%")
    return cell_meta







def mywknn(X_inputs, labels, is_pseudo_lb, k=10, metric='cosine', weights='distance', exclude_neighbor_num=1):
    """
    Implement a Weighted k-Nearest Neighbor classifier to refine pseudo-label predictions based on neighbor voting and distances.

    Parameters:
        X_inputs (array-like): Feature matrix (e.g., PCA embeddings).
        labels (array-like): Array of labels (including pseudo-labels).
        is_pseudo_lb (array-like): Boolean array indicating pseudo-label status.
        k (int): Number of neighbors to consider.
        metric (str): Distance metric ('cosine', 'euclidean', etc.).
        weights (str): Neighbor weighting strategy ('distance' or 'uniform').
        exclude_neighbor_num (int): Number of closest neighbors to exclude (e.g., self).

    Returns:
        tuple: Arrays (y_pred, y_pred_prob) with refined predictions and their probabilities.
    """
    X_train = X_inputs[is_pseudo_lb == 1]
    y_train = labels[is_pseudo_lb == 1]
    
    wknn = KNeighborsClassifier(n_neighbors=k + 1, weights=weights, metric=metric) 
    wknn.fit(X_train, y_train)

    distances, indices = wknn.kneighbors(X_train)

    y_pred = []
    y_pred_prob = []

    for i, neighbors in enumerate(indices):
        neighbors = neighbors[exclude_neighbor_num:]
        neighbor_labels = y_train[neighbors]

        neighbor_distances = distances[i][exclude_neighbor_num:] 
        if weights == 'distance':
            weight_values = 1 / (neighbor_distances + 1e-6) 
            bincounts = np.bincount(neighbor_labels, weights=weight_values)
        else:
            bincounts = np.bincount(neighbor_labels)

        pred_label = np.argmax(bincounts)
        y_pred.append(pred_label)

        pred_prob = bincounts[pred_label] / bincounts.sum()
        y_pred_prob.append(pred_prob)
    
    return np.array(y_pred), np.array(y_pred_prob)



def clean_pseudo_labels_with_wknn(X_inputs, all_labels, is_pseudo_lb, k=10, metric='cosine', weights='distance', exclude_neighbor_num=1, prob_threshold=0.5):
    """
    Clean pseudo-labels using Weighted k-Nearest Neighbor (WKNN) by keeping labels that match neighbor predictions with confidence above a threshold.

    Parameters:
        X_inputs (array-like): Feature matrix.
        all_labels (array-like): Original labels (including pseudo-labels).
        is_pseudo_lb (array-like): Boolean array indicating pseudo-label status.
        k (int): Number of neighbors.
        metric (str): Distance metric.
        weights (str): Neighbor weighting strategy.
        exclude_neighbor_num (int): Number of nearest neighbors to exclude.
        prob_threshold (float): Confidence threshold to retain pseudo-labels.

    Returns:
        array-like: Boolean array indicating cleaned pseudo-labels.
    """

    y_pred, y_pred_prob = mywknn(X_inputs=X_inputs, 
                                 labels=all_labels, 
                                 is_pseudo_lb=is_pseudo_lb, 
                                 k=k, 
                                 metric=metric, 
                                 weights=weights,
                                 exclude_neighbor_num=exclude_neighbor_num)

    cleaned_is_pseudo_lb = is_pseudo_lb.copy()

    pseudo_label_indices = np.where(is_pseudo_lb == 1)[0] 
    true_pseudo_labels = all_labels[pseudo_label_indices]

    cleaned_is_pseudo_lb[pseudo_label_indices] = np.logical_and(
        y_pred == true_pseudo_labels, y_pred_prob > prob_threshold
    ).astype(int)

    return cleaned_is_pseudo_lb





def seed_cleaning(adata, cell_meta, use_rep='X_pca', k=10):
    """
    Refine seed cells using WKNN-based pseudo-label cleaning and remove cell types with insufficient seeds.

    Parameters:
        adata (AnnData): AnnData object with dimensionality reduction embeddings.
        cell_meta (pd.DataFrame): Cell metadata including 'kendall_pred' and initial 'is_seed' labels.
        use_rep (str): Embedding key in adata.obsm (e.g., 'X_pca', 'lsi49').
        k (int): Number of neighbors for WKNN cleaning.

    Returns:
        pd.DataFrame: Updated cell_meta with refined 'is_seed' labels.
    """
    label_encoder = LabelEncoder()
    all_labels_encoded = label_encoder.fit_transform(cell_meta['kendall_pred'].values)
    cleaned_is_pseudo_lb = clean_pseudo_labels_with_wknn(
        X_inputs=adata.obsm[use_rep],
        all_labels=all_labels_encoded,
        is_pseudo_lb=cell_meta['is_seed'].values,
        k=k
    )
    cell_meta['is_seed'] = cleaned_is_pseudo_lb
    # exclude num <= 3 seeds
    category_counts = cell_meta['kendall_pred'][cell_meta['is_seed'] == 1].value_counts()
    small_categories = category_counts[category_counts <= 2].index
    cell_meta.loc[cell_meta['kendall_pred'].isin(small_categories), 'is_seed'] = 0

    return cell_meta





def atacannopy_wknn(adata, cell_meta, use_rep='lsi49', k=10):
    """
    Predict cell types for all cells using a Weighted k-Nearest Neighbor classifier trained on seed cells.

    Parameters:
        adata (AnnData): AnnData object containing embeddings.
        cell_meta (pd.DataFrame): Cell metadata indicating seed cells ('is_seed') and predicted types ('kendall_pred').
        use_rep (str): Embedding key in adata.obsm for training and prediction.
        k (int): Number of neighbors for WKNN prediction.

    Returns:
        pd.DataFrame: Updated cell_meta with WKNN-based predictions ('wknn_pred') and prediction probabilities ('wknn_pred_prob').
    """

    X_train = adata.obsm[use_rep][cell_meta['is_seed'] == 1]
    y_train = cell_meta['kendall_pred'][cell_meta['is_seed'] == 1].values
    X_test = adata.obsm[use_rep]

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    wknn_pred = knn.predict(X_test)
    wknn_pred_prob = knn.predict_proba(X_test)

    cell_meta['wknn_pred'] = wknn_pred
    cell_meta['wknn_pred_prob'] = wknn_pred_prob.max(axis=1)

    return cell_meta
