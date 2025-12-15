import numpy as np
import pandas as pd
import scipy
import scipy.spatial
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics
import sklearn.neighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, silhouette_samples

import torch

def jaccard_index(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.

    Parameters:
        set1 (set): First set of elements.
        set2 (set): Second set of elements.

    Returns:
        float: Jaccard similarity coefficient between set1 and set2.

    Example:
        score = jaccard(set([1,2,3]), set([2,3,4]))
    """

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union




def neighbor_conservation(orig_data, latent_data, n_neighbors=10, exclude_top=0):
    """
    Calculate biological conservation by comparing neighborhoods in original and latent spaces using average Jaccard similarity.

    Parameters:
        orig_data (np.ndarray): Original feature matrix (samples × features).
        latent_data (np.ndarray): Embedded feature matrix (samples × latent features).
        n_neighbors (int, optional): Number of nearest neighbors to compare. Default is 10.
        exclude_top (int, optional): Number of closest neighbors to exclude (e.g., self). Default is 0.

    Returns:
        float: Mean Jaccard similarity across all samples.
    """
    knn_orig = NearestNeighbors(n_neighbors=n_neighbors + exclude_top)
    knn_latent = NearestNeighbors(n_neighbors=n_neighbors + exclude_top)
    
    knn_orig.fit(orig_data)
    knn_latent.fit(latent_data)
    
    orig_neighbors = knn_orig.kneighbors(orig_data, return_distance=False)
    latent_neighbors = knn_latent.kneighbors(latent_data, return_distance=False)
    
    orig_neighbors = orig_neighbors[:, exclude_top:]
    latent_neighbors = latent_neighbors[:, exclude_top:]
    
    jaccard_scores = []
    for i in range(orig_data.shape[0]):
        orig_set = set(orig_neighbors[i])
        latent_set = set(latent_neighbors[i])
        
        score = jaccard_index(orig_set, latent_set)
        jaccard_scores.append(score)
    
    return np.mean(jaccard_scores)






def foscttm(x, y, device='cpu'):
    """
    Calculate the Fraction of Samples Closer Than True Match (FOSCTTM) to quantify integration accuracy between two embedding spaces.

    Parameters:
        x (np.ndarray or torch.Tensor): First embedding matrix (samples × features).
        y (np.ndarray or torch.Tensor): Second embedding matrix (samples × features).

    Returns:
        tuple: Arrays (foscttm_x, foscttm_y) representing the fraction of samples closer than the true match in both embeddings.
    """

    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")

    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        x_sq = torch.sum(x**2, dim=1)  # [N]
        y_sq = torch.sum(y**2, dim=1)  # [N]

        dist_sq = x_sq.unsqueeze(1) + y_sq.unsqueeze(0) - 2 * (x @ y.t())
        d = torch.sqrt(torch.clamp(dist_sq, min=0.0)) 
        diag_d = torch.diag(d)
        foscttm_x = (d < diag_d.unsqueeze(1)).float().mean(dim=1)
        foscttm_y = (d < diag_d.unsqueeze(0)).float().mean(dim=0)

    del x, y, x_sq, y_sq, dist_sq, d, diag_d
    if device != 'cpu':
        torch.cuda.empty_cache()

    return foscttm_x.cpu().numpy(), foscttm_y.cpu().numpy()


def partially_matched_foscttm(
    x: np.ndarray,
    y: np.ndarray,
    x_barcodes: np.ndarray,
    y_barcodes: np.ndarray
):
    """
    Compute FOSCTTM scores for partially matched data embeddings, quantifying how well corresponding cells align across two modalities.

    Parameters:
        x (np.ndarray): Embedding coordinates from dataset X (shape: N_x × D).
        y (np.ndarray): Embedding coordinates from dataset Y (shape: N_y × D).
        x_barcodes (np.ndarray): Cell barcodes from dataset X.
        y_barcodes (np.ndarray): Cell barcodes from dataset Y.

    Returns:
        tuple: (foscttm_x, foscttm_y), arrays indicating the fraction of cells closer than their true matches for each dataset.
    """

    x_barcode_to_index = {barcode: idx for idx, barcode in enumerate(x_barcodes)}
    y_barcode_to_index = {barcode: idx for idx, barcode in enumerate(y_barcodes)}
    
    matched_barcodes = np.intersect1d(x_barcodes, y_barcodes)

    if matched_barcodes.size == 0:
        raise ValueError("No matching barcodes found between x_barcodes and y_barcodes.")

    matched_indices_x = np.array([x_barcode_to_index[barcode] for barcode in matched_barcodes])
    matched_indices_y = np.array([y_barcode_to_index[barcode] for barcode in matched_barcodes])
    
    x_matched = x[matched_indices_x]
    y_matched = y[matched_indices_y]
    
    D_x = scipy.spatial.distance_matrix(x_matched, y)  # shape: (num_matched, N_y)
    D_y = scipy.spatial.distance_matrix(y_matched, x)  # shape: (num_matched, N_x)
    print(f"D_x.shape:{D_x.shape}")
    print(f"D_y.shape:{D_y.shape}")

    d_x_diag = D_x[np.arange(len(matched_indices_x)), matched_indices_y]
    d_y_diag = D_y[np.arange(len(matched_indices_y)), matched_indices_x]

    foscttm_x = (D_x < d_x_diag[:, np.newaxis]).mean(axis=1)
    foscttm_y = (D_y < d_y_diag[:, np.newaxis]).mean(axis=1)
    
    return foscttm_x, foscttm_y





def batch_ASW(latent, modality, celltype, verbose=False, **kwargs):
    """
    Compute batch effect removal effectiveness using the Adjusted Silhouette Width (ASW) per cell type.

    Parameters:
        latent (np.ndarray): Embedded representation of cells (samples × latent features).
        modality (array-like): Batch labels indicating different modalities or batches.
        celltype (array-like): Cell type labels for each cell.
        verbose (bool, optional): Whether to print additional information. Default is False.

    Returns:
        float: Mean ASW score reflecting batch mixing quality.
    """
    s_per_ct = []
    for t in np.unique(celltype):
        mask = celltype == t
        try:
            s = sklearn.metrics.silhouette_samples(latent[mask], modality[mask], **kwargs)
        except ValueError:  # Too few samples
            s = 0
        s = (1 - np.fabs(s)).mean()
        s_per_ct.append(s)
        if verbose:
            print(f"Cell type: {t}, Adjusted silhouette width: {s}")
    result = np.mean(s_per_ct)
    # Convert to float if it has .item() method (e.g., numpy scalar)
    if hasattr(result, 'item'):
        result = result.item()
    return result



def ct_ASW(latent, celltype, **kwargs):
    """
    Compute the Adjusted Silhouette Width (ASW) to evaluate how well cell types cluster in latent space.

    Parameters:
        latent (np.ndarray): Embedded feature matrix (samples × latent features).
        celltype (array-like): Cell type labels for each cell.

    Returns:
        float: Normalized cell-type ASW score between 0 and 1.
    """
    score = sklearn.metrics.silhouette_score(latent, celltype, **kwargs)
    # Convert to float if it has .item() method (e.g., numpy scalar), otherwise use as is
    if hasattr(score, 'item'):
        score = score.item()
    return (score + 1) / 2


def get_rare_cell_types(celltype, rare_threshold=0.02):
    """
    识别稀有细胞类型
    
    Parameters:
    -----------
    celltype : array-like
        所有细胞的类型标签
    rare_threshold : float or int
        如果是float (0-1): 占比阈值，例如0.02表示细胞数<2%的类型
        如果是int (>1): 绝对数量阈值，例如100表示细胞数<100的类型
    
    Returns:
    --------
    list
        稀有细胞类型列表
    """
    counts = pd.Series(celltype).value_counts()
    total = len(celltype)
    
    if rare_threshold < 1:  # 占比阈值
        rare_types = counts[counts / total < rare_threshold].index.tolist()
    else:  # 绝对数量阈值
        rare_types = counts[counts < rare_threshold].index.tolist()
    
    return rare_types


def score_rare_cell_type(latent, celltype, rare_type, scale=True):
    """
    计算单个稀有细胞类型的ASW得分
    
    复用scib的实现逻辑，使用sklearn.metrics.silhouette_samples
    
    Parameters:
    -----------
    latent : np.ndarray
        整合后的嵌入空间 (n_cells × n_features)
    celltype : array-like
        所有细胞的类型标签
    rare_type : str
        特定稀有细胞类型
    scale : bool
        是否归一化到[0,1]区间
    
    Returns:
    --------
    float
        该稀有类型的ASW得分
    """
    from sklearn.metrics import silhouette_samples
    
    # 计算所有细胞的silhouette score
    sil_scores = silhouette_samples(latent, celltype)
    
    # 只提取该稀有类型的得分并取平均
    mask = celltype == rare_type
    score = sil_scores[mask].mean()
    
    # 归一化到[0,1]区间
    if scale:
        score = (score + 1) / 2
    
    # Convert to float if needed
    if hasattr(score, 'item'):
        score = score.item()
    
    return score


def rare_ct_ASW(latent, celltype, rare_threshold=0.02, 
                scale=True, return_all=False, verbose=False):
    """
    Rare Cell Type ASW (RCT-ASW)
    
    评估稀有细胞类型在整合空间中的聚类质量
    参考scib.metrics.isolated_labels_asw的实现逻辑
    
    Parameters:
    -----------
    latent : np.ndarray
        整合后的嵌入空间 (n_cells × n_features)
    celltype : array-like
        所有细胞的类型标签（来自合并后的adata_cm）
    rare_threshold : float or int, default=0.02
        稀有类型的阈值
        - float (0-1): 占比阈值，默认2%
        - int (>1): 绝对数量阈值
    scale : bool, default=True
        是否归一化到[0,1]区间
    return_all : bool, default=False
        是否返回每个稀有类型的得分（dict），否则返回平均值（float）
    verbose : bool, default=False
        是否打印详细信息
    
    Returns:
    --------
    float or dict or None
        所有稀有类型的平均ASW得分，或每个稀有类型的得分字典
        如果没有稀有类型，返回None
    """
    # 识别稀有类型
    rare_types = get_rare_cell_types(celltype, rare_threshold)
    
    if verbose:
        print(f"    Found {len(rare_types)} rare cell types (threshold={rare_threshold}): {rare_types}")
    
    if len(rare_types) == 0:
        if verbose:
            print("    No rare cell types found!")
        return None
    
    # 计算每个稀有类型的ASW
    scores = {}
    for rare_type in rare_types:
        score = score_rare_cell_type(latent, celltype, rare_type, scale)
        scores[rare_type] = score
        if verbose:
            print(f"      {rare_type}: {score:.4f}")
    
    # 返回结果
    if return_all:
        return scores
    else:
        return np.mean(list(scores.values()))


def knn_matching(rna_latent, atac_latent, k=1):
    """
    Assess cross-modal matching accuracy between RNA and ATAC embeddings using k-Nearest Neighbors.

    Parameters:
        rna_latent (np.ndarray): RNA embedding matrix (cells × latent features).
        atac_latent (np.ndarray): ATAC embedding matrix (cells × latent features).
        k (int, optional): Number of neighbors to consider. Default is 1.

    Returns:
        tuple: Accuracy scores for RNA matching, ATAC matching, and overall accuracy.
    """
    similarity_matrix = cosine_similarity(rna_latent, atac_latent)

    # RNA -> ATAC
    rna_to_atac_neighbors = np.argsort(-similarity_matrix, axis=1)[:, :k]
    rna_matches = [i in rna_to_atac_neighbors[i] for i in range(len(rna_latent))]
    rna_correct_matches = np.sum(rna_matches)

    # ATAC -> RNA
    atac_to_rna_neighbors = np.argsort(-similarity_matrix.T, axis=1)[:, :k]
    atac_matches = [i in atac_to_rna_neighbors[i] for i in range(len(atac_latent))]
    atac_correct_matches = np.sum(atac_matches)

    rna_accuracy = rna_correct_matches / len(rna_latent)
    atac_accuracy = atac_correct_matches / len(atac_latent)
    overall_accuracy = (rna_correct_matches + atac_correct_matches) / (len(rna_latent) + len(atac_latent))

    return {
        "rna_to_atac_accuracy": np.round(rna_accuracy,5),
        "atac_to_rna_accuracy": np.round(atac_accuracy,5),
        "overall_accuracy": np.round(overall_accuracy,5)
    }




def cilisi(adata, batch_key, label_key, use_rep, k0=90,n_cores=10, scale=True, type_="embed", verbose=False):
    """
    Compute conditional iLISI (integration metric) per cell type, assessing integration quality conditioned on cell types.

    Parameters:
        adata (AnnData): AnnData object containing embeddings and metadata.
        batch_key (str): Key for modality labels in adata.obs (e.g., 'batch', 'modality').
        label_key (str): Key specifying cell type annotations.
        use_rep (str, optional): Embedding representation to use from adata.obsm. Default is 'X_pca'.
        k0 (int, optional): Number of nearest neighbors for ILISI calculation. Default is 90.
        n_cores (int, optional): Number of CPU cores for computation. Default is 10.
        scale (bool, optional): Whether to scale embeddings. Default is True.
        type_ (str, optional): Data type ('embed' or 'full'). Default is 'embed'.
        verbose (bool, optional): Print detailed output. Default is False.

    Returns:
        dict: ILISI scores per cell type indicating integration quality.
    """

    import scib

    cell_types = adata.obs[label_key].unique()
    ilisi_per_cell_type = []

    for cell_type in cell_types:
        subset_adata = adata[adata.obs[label_key] == cell_type]
        current_k0 = min(k0, int(subset_adata.shape[0]/2))
        if verbose:
            print(f"current_k0:{current_k0}")
        ilisi = scib.me.ilisi_graph(
            subset_adata, n_cores=n_cores, batch_key=batch_key, scale=scale, type_=type_, use_rep=use_rep,k0=current_k0, verbose=False
        )
        ilisi_per_cell_type.append(ilisi)
        if verbose:
            print(f"Cell type '{cell_type}': ilisi: {ilisi:.5f}")
    cilisi = np.nanmean(np.where(np.isfinite(ilisi_per_cell_type), ilisi_per_cell_type, np.nan))
    return cilisi




######### annotation benchmark functions #########


def get_benchmark(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    avg_recall = recall_score(true_labels, pred_labels, average='macro')
    avg_precision = precision_score(true_labels, pred_labels, average='macro')
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    out = {
        'accuracy': round(accuracy, 4),
        'average_recall': round(avg_recall, 4),
        'average_precision': round(avg_precision, 4),
        'macro_f1': round(macro_f1, 4)
    }
    return out



def get_each_recall(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    classes = np.unique(true_labels)
    recall_data = []

    for cls in classes:
        true_positives = np.sum((true_labels == cls) & (pred_labels == cls))
        true_count = np.sum(true_labels == cls)
        recall = true_positives / true_count if true_count != 0 else np.nan
        recall_data.append([true_positives, true_count, recall])

    # Overall recall
    overall_true_positives = np.sum(true_labels == pred_labels)
    overall_true_count = len(true_labels)
    overall_recall = overall_true_positives / overall_true_count

    recall_data.append([overall_true_positives, overall_true_count, overall_recall])
    recall_df = pd.DataFrame(recall_data, index=np.append(classes, 'all'), columns=['True Positives', 'True Count', 'Recall'])

    return recall_df



def get_each_precision(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    classes = np.unique(pred_labels)
    precision_data = []

    for cls in classes:
        true_positives = np.sum((true_labels == cls) & (pred_labels == cls))
        pred_count = np.sum(pred_labels == cls)
        precision = true_positives / pred_count if pred_count != 0 else 0
        precision_data.append([true_positives, pred_count, precision])

    # Overall precision
    overall_true_positives = np.sum(true_labels == pred_labels)
    overall_pred_count = len(pred_labels)
    overall_precision = overall_true_positives / overall_pred_count

    precision_data.append([overall_true_positives, overall_pred_count, overall_precision])
    precision_df = pd.DataFrame(precision_data, index=np.append(classes, 'all'), columns=['True Positives', 'Pred Count', 'Precision'])

    return precision_df






def get_merged_labels_dataset_ISSAACseq(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "R0 Ex-L2/3 IT": "R0",
        "R1 Ex-L2/3 IT Act": "R1",
        "R10 Ex-L6b": "R10",
        "R11 Ex-PIR Ndst4": "R11",
        "R13 In-Drd2": "R13",
        "R14 In-Hap1": "R14",
        "R15 In-Pvalb": "R15",
        "R16 In-Sst": "R16",
        "R17 In-Tac1": "R17",
        "R18 In-Vip/Lamp5": "R18",
        "R19 Astro": "R19",
        "R2 Ex-L4 IT": "R2",
        "R20 OPC": "R20",
        "R21 Oligo": "R21",
        "R22 VLMC": "R22",
        "R3 Ex-L5 IT": "R3",
        "R4 Ex-L5 NP": "R4",
        "R5 Ex-L5 NP Cxcl14": "R5",
        "R6 Ex-L5-PT": "R6",
        "R7 Ex-L6 CT": "R7",
        "R8 Ex-L6 IT Bmp3": "R8",
        "R9 Ex-L6 IT Oprk1": "R9"
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_dataset_SHAREseq(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "ahighCD34+ bulge": 'HCD34B',
        "alowCD34+ bulge": 'LCD34B',
        "Basal": 'Bas',
        "Dermal Fibroblast": 'DF',
        "Dermal Papilla": 'DP',
        "Dermal Sheath": 'DS',
        "Endothelial": 'Endo',
        "Granular": 'Gran',
        "Hair Shaft-cuticle.cortex": 'HSCC',
        "Infundibulum": 'Infu',
        "IRS": 'IRS',
        "Isthmus": 'Isth',
        "K6+ Bulge Companion Layer": 'KBCL',
        "Macrophage DC": 'MDC',
        "Medulla": 'Medu',
        "Melanocyte": 'Mela',
        "ORS": 'ORS',
        "Schwann Cell": 'SC',
        "Sebaceous Gland": 'SG',
        "Spinous": 'Spin',
        "TAC-1": 'TAC1',
        "TAC-2": 'TAC2'
    }
    return [label_map.get(label, label) for label in labels]



def get_merged_labels_Kidney(labels):
    labels = [str(label) for label in labels]
    label_map = {
        'PCT': 'PT',
        'PST': 'PT',
        'DCT1': 'DCT',
        'DCT2': 'DCT',
        'MES': 'MES/FIB',
        'FIB': 'MES/FIB',
        'MES_FIB': 'MES/FIB'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Zhu(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD16 Mono": 'Mono', 
        "CD14 Mono": 'Mono', 
        "Monocytes": 'Mono',
        "cDC": 'DC', 
        "pDC": 'DC', 
        "DCs": 'DC',
        "CD4 Naive": 'NaiveT', 
        "CD8 Naive": 'NaiveT', 
        "Naive T cells": 'NaiveT',
        "CD4 TCM": 'CD4T', 
        "Treg": 'CD4T', 
        "CD4 TEM": 'CD4T', 
        "Activated CD4 T cells": 'CD4T',
        "CD8 TEM_2": 'CD8T', 
        "CD8 TEM_1": 'CD8T', 
        "Cytotoxic CD8 T cells": 'CD8T',
        "NK": 'ILC', 
        "NKs": 'ILC', 
        "XCL+ NKs": 'ILC',
        "Memory B": 'MemB', 
        "Memory B cells": 'MemB', 
        "Intermediate B": 'MemB',
        "Naive B": 'NaiveB', 
        "Naive B cells": 'NaiveB',
        "Plasma": 'Plasma', 
        "Cycling Plasma": 'Plasma',
        "Cycling T cells": 'CycT',
        "Megakaryocytes": 'Mega',
        "Stem cells": 'HSPC'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Wilk(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD16 Mono": 'CD16Mono', "CD16 Monocyte": 'CD16Mono',
        "CD14 Mono": 'CD14Mono', "CD14 Monocyte": 'CD14Mono',
        "cDC": 'cDC', "DC": 'cDC',
        "NK": 'ILC',
        "CD4 Naive": 'NaiveT', "CD8 Naive": 'NaiveT', "CD4n T": 'NaiveT',
        "CD4 TCM": 'CD4T', "Treg": 'CD4T', "CD4 TEM": 'CD4T', "CD4m T": 'CD4T', "CD4 T": 'CD4T',
        "CD8m T": 'CD8T', "CD8 TEM_2": 'CD8T', "CD8 TEM_1": 'CD8T', "MAIT": 'CD8T', "CD8eff T": 'CD8T',
        "gdT": 'gdT', "gd T": 'gdT',
        "Intermediate B": 'B', "Memory B": 'B', "Naive B": 'B', "B": 'B',
        "Plasmablast": 'Plasma', "Plasma": 'Plasma',
        "SC & Eosinophil": 'HSPC', "HSPC": 'HSPC',
        "Granulocyte": 'Granu'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Stephenson(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD16 Mono": 'CD16Mono', "CD16.mono": 'CD16Mono',
        "CD14 Mono": 'CD14Mono', "CD14.mono": 'CD14Mono',
        "cDC": 'cDC', "DC": 'cDC',
        "CD4 Naive": 'CD4NT', "CD4.Naive": 'CD4NT',
        "CD8 Naive": 'CD8NT', "CD8.Naive": 'CD8NT',
        "CD4 TCM": 'CD4T', "CD4 TEM": 'CD4T', "CD4.CM": 'CD4T', "CD4.IL22": 'CD4T',
        "CD4.Th": 'CD4T', "CD4.EM": 'CD4T', "CD4.Tfh": 'CD4T',
        "CD8.TE": 'CD8T', "CD8.EM": 'CD8T', "CD8 TEM_2": 'CD8T', "CD8 TEM_1": 'CD8T',
        "Intermediate B": 'MemB', "Memory B": 'MemB', "B_non-switched_memory": 'MemB',
        "B_switched_memory": 'MemB', "B_exhausted": 'MemB',
        "B_naive": 'NaiveB', "Naive B": 'NaiveB', "B_immature": 'NaiveB',
        "HSC": 'HSPC', "HSPC": 'HSPC',
        "NK": 'ILC', "ILC": 'ILC',
        "Lymph.prolif": 'Prolif',
        "Platelets": 'Platelet'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Hao(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD16 Mono": 'CD16Mono',
        "CD14 Mono": 'CD14Mono',
        "CD4 Naive": 'CD4NT',
        "CD8 Naive": 'CD8NT',
        "CD4 CTL": 'CD4CTL',
        "CD8 TEM": 'CD8T', "CD8 TCM": 'CD8T', "CD8 TEM_2": 'CD8T', "CD8 TEM_1": 'CD8T',
        "CD4 TEM": 'CD4T', "CD4 TCM": 'CD4T', "CD4 CTL": 'CD4T',
        "Intermediate B": 'InterB', "B intermediate": 'InterB',
        "Memory B": 'MemB', "B memory": 'MemB',
        "Naive B": 'NaiveB', "B naive": 'NaiveB',
        "Plasmablast": 'Plasma', "Plasma": 'Plasma',
        "NK": 'ILC', "ILC": 'ILC',
        "pDC": 'pDC', "ASDC": 'pDC',
        "Proliferating": 'Prolif'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Monaco(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD16 Mono": 'CD16Mono', "NC_mono": 'CD16Mono',
        "CD14 Mono": 'CD14Mono', "C_mono": 'CD14Mono',
        "I_mono": 'InterMono',
        "NK": 'ILC',
        "cDC": 'cDC', "mDC": 'cDC',
        "CD4 Naive": 'CD4NT', "CD4_naive": 'CD4NT',
        "CD8 Naive": 'CD8NT', "CD8_naive": 'CD8NT',
        "CD4 TCM": 'CD4T', "CD4 TEM": 'CD4T', "CD4_TE": 'CD4T', "TFH": 'CD4T',
        "Th1": 'CD4T', "Th1.Th17": 'CD4T', "Th17": 'CD4T', "Th2": 'CD4T', "Th1/Th17": 'CD4T',
        "CD8 TEM_2": 'CD8T', "CD8 TEM_1": 'CD8T', "CD8_CM": 'CD8T', "CD8_EM": 'CD8T', "CD8_TE": 'CD8T',
        "gdT": 'gdT', "VD2-": 'gdT', "VD2+": 'gdT', "VD2_gdT": 'gdT', "nVD2_gdT": 'gdT',
        "Intermediate B": 'MemB', "Memory B": 'MemB', "B_NSM": 'MemB', "B_Ex": 'MemB', "B_SM": 'MemB',
        "Naive B": 'NaiveB', "B_naive": 'NaiveB',
        "Plasmablasts": 'Plasma', "Plasma": 'Plasma',
        "Progenitor": 'HSPC', "HSPC": 'HSPC'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_10XMultiome(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "CD14 Mono": 'CD14Mono',
        "CD16 Mono": 'CD16Mono',
        "CD4 Naive": 'CD4NT',
        "CD8 Naive": 'CD8NT',
        "CD4 TCM": 'CD4T', "CD4 TEM": 'CD4T',
        "NK": 'ILC',
        "Naive B": 'NaiveB',
        "CD8 TEM_2": 'CD8T', "CD8 TEM_1": 'CD8T',
        "Memory B": 'MemB',
        "Intermediate B": 'InterB'
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_MouseKid(labels):
    labels = [str(label) for label in labels]
    label_map = {
        'Stroma 1': 'Stroma', 'Stroma 2': 'Stroma',
        'Early PT': 'PST', 'PST': 'PST'
    }
    return [label_map.get(label, label) for label in labels]


def get_merged_labels_histone(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "Astrocytes": "Astro",
        "mOL": "Oligo",
        "Endothelial": "VLMC",
        "Mural": "VLMC",
        "Neurons_1": "Neuron",
        "Neurons_2": "Neuron",
        "Neurons_3": "Neuron"
    }
    return [label_map.get(label, label) for label in labels]


def get_merged_labels_pancreas(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "activated_stellate": "mesenchymal",
        "quiescent_stellate": "mesenchymal"
    }
    return [label_map.get(label, label) for label in labels]



def get_merged_labels_LungDroplet(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "Myeloid cell": "Myeloid cell",
        "Dendritic cell": "Myeloid cell",
        "Macrophage": "Myeloid cell",
        "Monocyte": "Myeloid cell",
        "Endothelial cell": "Stromal cell",
        "Stromal cell": "Stromal cell"
    }
    return [label_map.get(label, label) for label in labels]

def get_merged_labels_Brain(labels):
    labels = [str(label) for label in labels]
    label_map = {
        "ASC": "Astrocytes",
        "EX": "Excitatory",
        "INH": "Inhibitory",
        "MG": "Microglia",
        "ODC": "Oligodendrocytes",
        "OPC": "OPC",
        "PER.END": "Endothelial"
    }
    return [label_map.get(label, label) for label in labels]



def GET_GML(dataset):
    if dataset == 'ISSAAC-seq' or dataset.lower() == 'issaac':
        return get_merged_labels_dataset_ISSAACseq
    elif dataset == 'SHARE-seq' or dataset.lower() == 'share':
        return get_merged_labels_dataset_SHAREseq
    elif dataset.lower() == 'kidney':
        return get_merged_labels_Kidney
    elif dataset.lower() == 'zhu':
        return get_merged_labels_Zhu
    elif dataset.lower() == 'wilk':
        return get_merged_labels_Wilk
    elif dataset == 'Stephenson':
        return get_merged_labels_Stephenson
    elif dataset == 'Hao':
        return get_merged_labels_Hao
    elif dataset == 'Monaco':
        return get_merged_labels_Monaco
    elif dataset == 'LungDroplet':
        return get_merged_labels_LungDroplet
    # elif dataset.lower() == 'brain':
    #     return get_merged_labels_Brain
    elif dataset == 'MouseKid':
        return get_merged_labels_MouseKid
    elif dataset == 'histone':
        return get_merged_labels_histone
    elif dataset == 'pancreas':
        return get_merged_labels_pancreas
    else:
        return lambda x: x






