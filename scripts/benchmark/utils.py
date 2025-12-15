"""
Benchmark 评估工具函数
提供共享的数据加载、指标计算和结果保存功能
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scib

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scmiac.evaluation.benchmark import (
    neighbor_conservation, foscttm, batch_ASW, ct_ASW, cilisi, GET_GML, rare_ct_ASW
)

# 配对数据集列表
PAIRED_DATASETS = ['10x', 'share']


def check_embeddings_exist(rna_h5ad_path, atac_h5ad_path, rna_emb_path, atac_emb_path):
    """
    检查评估所需的文件是否存在
    
    Parameters:
    -----------
    rna_h5ad_path : Path or str
        RNA 原始数据文件路径
    atac_h5ad_path : Path or str
        ATAC 原始数据文件路径
    rna_emb_path : Path or str
        RNA 嵌入文件路径
    atac_emb_path : Path or str
        ATAC 嵌入文件路径
        
    Returns:
    --------
    bool
        所有文件都存在返回 True，否则返回 False
    """
    paths = [rna_h5ad_path, atac_h5ad_path, rna_emb_path, atac_emb_path]
    return all(Path(p).exists() for p in paths)


def load_benchmark_data(dataset, rna_h5ad_path, atac_h5ad_path, rna_emb_path, atac_emb_path):
    """
    加载评估所需的原始数据和嵌入
    
    Parameters:
    -----------
    dataset : str
        数据集名称（用于获取细胞类型合并函数）
    rna_h5ad_path : Path or str
        RNA 原始数据文件路径
    atac_h5ad_path : Path or str
        ATAC 原始数据文件路径
    rna_emb_path : Path or str
        RNA 嵌入文件路径（支持 .npy 或 .csv 格式）
    atac_emb_path : Path or str
        ATAC 嵌入文件路径（支持 .npy 或 .csv 格式）
        
    Returns:
    --------
    tuple
        (adata_rna, adata_atac, rna_emb, atac_emb)
    """
    # 加载原始数据
    adata_rna = sc.read_h5ad(rna_h5ad_path)
    adata_atac = sc.read_h5ad(atac_h5ad_path)
    
    # 统一细胞类型列名
    if 'celltype' in adata_rna.obs.columns:
        adata_rna.obs['cell_type'] = adata_rna.obs['celltype']
    if 'celltype' in adata_atac.obs.columns:
        adata_atac.obs['cell_type'] = adata_atac.obs['celltype']
    
    # 添加 cell_type_merge
    GML = GET_GML(dataset)
    adata_rna.obs['cell_type_merge'] = GML(adata_rna.obs['cell_type'])
    adata_atac.obs['cell_type_merge'] = GML(adata_atac.obs['cell_type'])
    
    # 加载嵌入（支持 .npy 和 .csv 格式）
    def load_embedding(path):
        path = Path(path)
        if path.suffix == '.npy':
            return np.load(path)
        elif path.suffix == '.csv':
            # CSV 格式：第一列是索引（cell ID），其余列是嵌入维度
            df = pd.read_csv(path, index_col=0)
            return df.values
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    rna_emb = load_embedding(rna_emb_path)
    atac_emb = load_embedding(atac_emb_path)
    
    return adata_rna, adata_atac, rna_emb, atac_emb


def load_benchmark_data_vertical(dataset, rna_h5ad_path, atac_h5ad_path, 
                                  rna_emb_path, atac_emb_path, paired_cells):
    """
    加载垂直整合评估所需的数据
    
    与标准加载的区别：
    1. 只保留未配对细胞（用于评估）
    2. 处理带 -RNA/-ATAC 后缀的细胞名
    3. 将嵌入中的细胞映射回原始细胞名
    
    Parameters:
    -----------
    dataset : str
        数据集名称
    rna_h5ad_path : Path or str
        RNA 原始数据文件路径
    atac_h5ad_path : Path or str
        ATAC 原始数据文件路径
    rna_emb_path : Path or str
        RNA 嵌入文件路径
    atac_emb_path : Path or str
        ATAC 嵌入文件路径
    paired_cells : set
        配对细胞集合
        
    Returns:
    --------
    tuple
        (adata_rna_unpaired, adata_atac_unpaired, rna_emb, atac_emb)
    """
    # 加载原始数据
    adata_rna = sc.read_h5ad(rna_h5ad_path)
    adata_atac = sc.read_h5ad(atac_h5ad_path)
    
    # 统一细胞类型列名
    if 'celltype' in adata_rna.obs.columns:
        adata_rna.obs['cell_type'] = adata_rna.obs['celltype']
    if 'celltype' in adata_atac.obs.columns:
        adata_atac.obs['cell_type'] = adata_atac.obs['celltype']
    
    # 添加 cell_type_merge
    GML = GET_GML(dataset)
    adata_rna.obs['cell_type_merge'] = GML(adata_rna.obs['cell_type'])
    adata_atac.obs['cell_type_merge'] = GML(adata_atac.obs['cell_type'])
    
    # 过滤出未配对细胞
    unpaired_cells = [c for c in adata_rna.obs_names if c not in paired_cells]
    adata_rna_unpaired = adata_rna[unpaired_cells].copy()
    adata_atac_unpaired = adata_atac[unpaired_cells].copy()
    
    # 加载嵌入
    def load_embedding(path):
        path = Path(path)
        if path.suffix == '.csv':
            df = pd.read_csv(path, index_col=0)
            return df
        else:
            raise ValueError(f"Vertical benchmark only supports .csv format, got: {path.suffix}")
    
    rna_emb_df = load_embedding(rna_emb_path)
    atac_emb_df = load_embedding(atac_emb_path)
    
    # 处理带后缀的细胞名：去掉 -RNA/-ATAC 后缀
    rna_emb_df.index = rna_emb_df.index.str.replace('-RNA$', '', regex=True)
    atac_emb_df.index = atac_emb_df.index.str.replace('-ATAC$', '', regex=True)
    
    # 确保顺序与 adata 一致
    rna_emb = rna_emb_df.loc[adata_rna_unpaired.obs_names].values
    atac_emb = atac_emb_df.loc[adata_atac_unpaired.obs_names].values
    
    return adata_rna_unpaired, adata_atac_unpaired, rna_emb, atac_emb


def compute_all_metrics(adata_rna, adata_atac, rna_emb, atac_emb, 
                        dataset, method_name='temp',
                        n_neighbors=10, n_cores=10, verbose=False):
    """
    计算所有评估指标
    
    Parameters:
    -----------
    adata_rna : AnnData
        RNA 原始数据
    adata_atac : AnnData
        ATAC 原始数据
    rna_emb : ndarray
        RNA 嵌入
    atac_emb : ndarray
        ATAC 嵌入
    dataset : str
        数据集名称
    method_name : str
        方法名称（用于 obsm key）
    n_neighbors : int
        邻居数量
    n_cores : int
        并行核心数
    verbose : bool
        是否显示详细信息
        
    Returns:
    --------
    dict
        包含所有指标的字典
    """
    # 创建合并数据
    adata_rna.obs['modality'] = 'RNA'
    adata_atac.obs['modality'] = 'ATAC'
    adata_rna.obsm[f'{method_name}_latent'] = rna_emb
    adata_atac.obsm[f'{method_name}_latent'] = atac_emb
    adata_cm = sc.concat([adata_rna, adata_atac], join='outer')
    
    # 确保关键列是 category 类型（scib 要求）
    if not pd.api.types.is_categorical_dtype(adata_cm.obs['modality']):
        adata_cm.obs['modality'] = adata_cm.obs['modality'].astype('category')
    if not pd.api.types.is_categorical_dtype(adata_cm.obs['cell_type_merge']):
        adata_cm.obs['cell_type_merge'] = adata_cm.obs['cell_type_merge'].astype('category')
    
    results = {}
    latent = adata_cm.obsm[f'{method_name}_latent']
    
    # LISI 指标
    if verbose:
        print("    Computing LISI...")
    try:
        results['ilisi'] = scib.me.ilisi_graph(
            adata_cm, batch_key="modality", scale=True,
            type_="embed", use_rep=f'{method_name}_latent', n_cores=n_cores, verbose=False
        )
    except Exception as e:
        if verbose:
            print(f"      iLISI failed: {e}")
        results['ilisi'] = None
    
    try:
        results['clisi'] = scib.me.clisi_graph(
            adata_cm, label_key="cell_type_merge", scale=True,
            type_="embed", use_rep=f'{method_name}_latent', n_cores=n_cores, verbose=False
        )
    except Exception as e:
        if verbose:
            print(f"      cLISI failed: {e}")
        results['clisi'] = None
    
    try:
        results['cilisi'] = cilisi(
            adata_cm, batch_key="modality", label_key="cell_type_merge",
            use_rep=f'{method_name}_latent', n_cores=n_cores, scale=True,
            type_="embed", verbose=False
        )
    except Exception as e:
        if verbose:
            print(f"      CiLISI failed: {e}")
        results['cilisi'] = None
    
    # ASW 指标
    if verbose:
        print("    Computing ASW...")
    try:
        results['batch_asw'] = batch_ASW(
            latent, adata_cm.obs['modality'].values,
            adata_cm.obs['cell_type_merge'].values
        )
    except Exception as e:
        if verbose:
            print(f"      Batch ASW failed: {e}")
        results['batch_asw'] = None
    
    try:
        results['ct_asw'] = ct_ASW(latent, adata_cm.obs['cell_type_merge'].values)
    except Exception as e:
        if verbose:
            print(f"      CT ASW failed: {e}")
        results['ct_asw'] = None
    
    # Rare Cell Type ASW
    if verbose:
        print("    Computing Rare CT ASW...")
    try:
        results['rct_asw'] = rare_ct_ASW(
            latent, 
            adata_cm.obs['cell_type_merge'].values,
            rare_threshold=0.02,
            scale=True,
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"      Rare CT ASW failed: {e}")
        results['rct_asw'] = None
    
    # Isolated Label ASW
    if verbose:
        print("    Computing ilASW...")
    try:
        results['ilasw'] = scib.me.isolated_labels_asw(
            adata_cm,
            batch_key="modality",
            label_key="cell_type_merge",
            embed=f'{method_name}_latent',
            iso_threshold=1,
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"      ilASW failed: {e}")
        results['ilasw'] = None
    
    # Neighbor Conservation
    if verbose:
        print("    Computing NC...")
    rna_mask = adata_cm.obs['modality'] == 'RNA'
    atac_mask = adata_cm.obs['modality'] == 'ATAC'
    
    rna_orig = adata_rna.obsm.get('X_pca', adata_rna.X)
    atac_orig = adata_atac.obsm.get('lsi49', adata_atac.X)
    
    try:
        results['rna_nc'] = neighbor_conservation(
            rna_orig if hasattr(rna_orig, 'shape') else rna_orig.toarray(),
            latent[rna_mask], n_neighbors=n_neighbors
        )
    except Exception as e:
        if verbose:
            print(f"      RNA NC failed: {e}")
        results['rna_nc'] = None
    
    try:
        results['atac_nc'] = neighbor_conservation(
            atac_orig if hasattr(atac_orig, 'shape') else atac_orig.toarray(),
            latent[atac_mask], n_neighbors=n_neighbors
        )
    except Exception as e:
        if verbose:
            print(f"      ATAC NC failed: {e}")
        results['atac_nc'] = None
    
    # 计算平均 NC（合并 RNA 和 ATAC）
    if results['rna_nc'] is not None and results['atac_nc'] is not None:
        results['nc'] = (results['rna_nc'] + results['atac_nc']) / 2
    elif results['rna_nc'] is not None:
        results['nc'] = results['rna_nc']
    elif results['atac_nc'] is not None:
        results['nc'] = results['atac_nc']
    else:
        results['nc'] = None
    
    # Graph Connectivity
    if verbose:
        print("    Computing GC...")
    try:
        sc.pp.neighbors(adata_cm, use_rep=f"{method_name}_latent")
        results['graph_connectivity'] = scib.me.graph_connectivity(
            adata_cm, label_key="cell_type_merge"
        )
    except Exception as e:
        if verbose:
            print(f"      GC failed: {e}")
        results['graph_connectivity'] = None
    
    # 配对数据集专属指标
    if dataset in PAIRED_DATASETS:
        if verbose:
            print("    Computing paired metrics...")
        # Neighbor Consistency (在整合空间中计算)
        try:
            results['neighbor_consistency'] = neighbor_conservation(
                rna_emb,
                atac_emb,
                n_neighbors=n_neighbors
            )
        except Exception as e:
            if verbose:
                print(f"      Neighbor consistency failed: {e}")
            results['neighbor_consistency'] = None
        
        # 原始空间的 Neighbor Consistency (作为参考)
        try:
            results['neighbor_consistency_orig'] = neighbor_conservation(
                rna_orig if hasattr(rna_orig, 'shape') else rna_orig.toarray(),
                atac_orig if hasattr(atac_orig, 'shape') else atac_orig.toarray(),
                n_neighbors=n_neighbors
            )
        except Exception as e:
            if verbose:
                print(f"      Original space neighbor consistency failed: {e}")
            results['neighbor_consistency_orig'] = None
        
        # FOSCTTM
        try:
            foscttm_result = foscttm(rna_emb, atac_emb, device='cpu')
            results['foscttm_rna'] = np.mean(foscttm_result[0])
            results['foscttm_atac'] = np.mean(foscttm_result[1])
            # 计算平均 FOSCTTM（合并 RNA 和 ATAC）
            results['foscttm'] = (results['foscttm_rna'] + results['foscttm_atac']) / 2
        except Exception as e:
            if verbose:
                print(f"      FOSCTTM failed: {e}")
            results['foscttm_rna'] = None
            results['foscttm_atac'] = None
            results['foscttm'] = None
    
    return results


def save_benchmark_summary(output_path, results_dict):
    """
    保存评估结果到 CSV 文件
    
    Parameters:
    -----------
    output_path : Path or str
        输出 CSV 文件路径
    results_dict : dict
        评估结果字典，格式为 {method_name: {metric: value}}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results_dict).T
    df.index.name = 'method'
    df.to_csv(output_path, float_format='%.5f')
    
    print(f"\n  ✓ Summary saved to {output_path}")
    print(f"\n{df}")
