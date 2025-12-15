#!/usr/bin/env python
# PAMONA 方法运行脚本
"""
运行 PAMONA 方法进行多模态整合，支持细胞类型先验信息

Usage:
    python pamona.py --dataset 10x --output-dir data/10x/output/methods/pamona
    python pamona.py --dataset 10x --output-dir data/10x/output/methods/pamona_noct --no-prior
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pamona
    from pamona import Pamona
except ImportError:
    print("Error: pamona not installed. Please install it first.", file=sys.stderr)
    sys.exit(1)


def set_random_seed(seed: int = 24):
    """设置随机种子"""
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data(dataset_name: str, data_root: str = "data"):
    """加载预处理的 RNA 和 ATAC 数据"""
    print(f"Loading {dataset_name} dataset...")
    
    data_dir = Path(data_root) / dataset_name / "input"
    
    rna_path = data_dir / f"adata_rna_{dataset_name}.h5ad"
    atac_path = data_dir / f"adata_atac_{dataset_name}.h5ad"
    
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA data not found: {rna_path}")
    if not atac_path.exists():
        raise FileNotFoundError(f"ATAC data not found: {atac_path}")
    
    adata_rna = sc.read(rna_path)
    adata_atac = sc.read(atac_path)
    
    print(f"RNA: {adata_rna.shape}, ATAC: {adata_atac.shape}")
    return adata_rna, adata_atac


def prepare_data(adata_rna, adata_atac, rna_celltype_key='cell_type', atac_celltype_key='pred'):
    """准备 PAMONA 输入数据
    
    Args:
        adata_rna: RNA AnnData对象
        adata_atac: ATAC AnnData对象
        rna_celltype_key: RNA细胞类型列名 (default: 'cell_type')
        atac_celltype_key: ATAC细胞类型列名 (default: 'pred')
    
    Returns:
        data: 标准化后的数据列表
        datatype: 编码后的细胞类型标签列表（如果可用）
    """
    print("Preparing data for PAMONA...")
    print(f"Using RNA cell type key: '{rna_celltype_key}'")
    print(f"Using ATAC cell type key: '{atac_celltype_key}'")
    
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    
    if 'X_pca' not in adata_rna.obsm:
        print("Computing PCA for RNA...")
        sc.pp.pca(adata_rna)
    RNA_arr = adata_rna.obsm['X_pca']
    
    if 'lsi49' in adata_atac.obsm:
        ATAC_arr = adata_atac.obsm['lsi49']
    elif 'X_lsi' in adata_atac.obsm:
        ATAC_arr = adata_atac.obsm['X_lsi']
    else:
        print("Computing LSI for ATAC...")
        sc.pp.pca(adata_atac)
        ATAC_arr = adata_atac.obsm['X_pca']
    
    # 检查 RNA 细胞类型（硬依赖）
    if rna_celltype_key not in adata_rna.obs.columns:
        raise KeyError(
            f"RNA cell type column '{rna_celltype_key}' not found in adata_rna.obs. "
            f"Available columns: {list(adata_rna.obs.columns)}. "
            f"PAMONA requires cell type annotations for both modalities to ensure fair comparison. "
            f"Use --rna-celltype-key to specify the correct column name."
        )
    
    # 检查 ATAC 细胞类型（硬依赖）
    if atac_celltype_key not in adata_atac.obs.columns:
        raise KeyError(
            f"ATAC cell type column '{atac_celltype_key}' not found in adata_atac.obs. "
            f"Available columns: {list(adata_atac.obs.columns)}. "
            f"PAMONA requires cell type annotations for both modalities to ensure fair comparison. "
            f"Use --atac-celltype-key to specify the correct column name."
        )
    
    # 获取细胞类型标签
    RNA_labels = adata_rna.obs[rna_celltype_key].values
    ATAC_labels = adata_atac.obs[atac_celltype_key].values
    
    RNA_encoded_labels = None
    ATAC_encoded_labels = None
    
    if RNA_labels is not None and ATAC_labels is not None:
        RNA_encoded_labels = label_encoder.fit_transform(RNA_labels)
        ATAC_encoded_labels = label_encoder.transform(ATAC_labels)
    
    data1 = pamona.utils.zscore_standardize(np.asarray(RNA_arr))
    data2 = pamona.utils.zscore_standardize(np.asarray(ATAC_arr))
    
    data = [data1, data2]
    datatype = None
    
    if RNA_encoded_labels is not None and ATAC_encoded_labels is not None:
        RNA_encoded_labels = RNA_encoded_labels.astype(int)
        ATAC_encoded_labels = ATAC_encoded_labels.astype(int)
        datatype = [RNA_encoded_labels, ATAC_encoded_labels]
    
    return data, datatype


def create_prior_matrix(data, datatype, use_prior: bool = True):
    """创建先验矩阵 M（使用细胞类型信息）"""
    # datatype 应该始终存在（硬依赖已在 prepare_data 中检查）
    if datatype is None:
        raise ValueError("datatype is None. Cell type information is required for PAMONA.")
    
    # 如果用户主动禁用先验，返回 None 但不报错
    if not use_prior:
        print("Prior information disabled by user (--no-prior)")
        return None
    
    print("Creating prior matrix M with cell type information...")
    M = []
    n_datasets = len(data)
    
    for k in range(n_datasets - 1):
        M_k = np.ones((len(data[k]), len(data[-1])))
        for i in range(len(data[k])):
            for j in range(len(data[-1])):
                if datatype[k][i] == datatype[-1][j]:
                    M_k[i][j] = 0.5
        M.append(M_k)
    
    print(f"Prior matrix M created with shape: {[m.shape for m in M]}")
    return M


def run_pamona(data, M=None):
    """运行 PAMONA 整合"""
    mode = "with cell type prior" if M is not None else "without prior"
    print(f"Running PAMONA integration ({mode})...")
    
    Pa = Pamona.Pamona(M=M) if M is not None else Pamona.Pamona()
    integrated_data, T = Pa.run_Pamona(data)
    
    print("PAMONA integration completed")
    return integrated_data


def save_results(adata_rna, adata_atac, integrated_data, output_dir: Path):
    """保存潜在表示"""
    print("Saving latent representations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rna_embeddings = integrated_data[0]
    atac_embeddings = integrated_data[1]
    
    adata_rna.obsm['pamona_latent'] = rna_embeddings
    adata_atac.obsm['pamona_latent'] = atac_embeddings
    
    # 保存为 CSV 格式
    rna_embeddings_csv = output_dir / "rna_embeddings.csv"
    atac_embeddings_csv = output_dir / "atac_embeddings.csv"
    
    pd.DataFrame(rna_embeddings, index=adata_rna.obs_names).to_csv(rna_embeddings_csv)
    pd.DataFrame(atac_embeddings, index=adata_atac.obs_names).to_csv(atac_embeddings_csv)
    
    print(f"RNA embeddings saved to {rna_embeddings_csv}")
    print(f"ATAC embeddings saved to {atac_embeddings_csv}")
    
    return adata_rna, adata_atac


def generate_umap(adata_rna, adata_atac, output_dir: Path):
    """生成 UMAP 可视化"""
    print("Generating UMAP visualization...")
    
    adata_rna.obs['modality'] = "RNA"
    adata_atac.obs['modality'] = "ATAC"
    
    adata_cm = anndata.concat([adata_rna, adata_atac], join='outer')
    
    sc.pp.neighbors(adata_cm, use_rep='pamona_latent', key_added='pamona')
    sc.tl.umap(adata_cm, neighbors_key='pamona')
    adata_cm.obsm['pamona_latent_umap'] = adata_cm.obsm['X_umap']
    
    # 保存 UMAP 图片
    umap_fig_path = output_dir / "pamona_latent_umap.png"
    color_list = ["cell_type", "modality"] if "cell_type" in adata_cm.obs else ["modality"]
    fig = sc.pl.embedding(
        adata_cm,
        basis='X_umap',
        color=color_list,
        legend_loc="on data",
        show=False,
        return_fig=True
    )
    fig.savefig(umap_fig_path, bbox_inches='tight', dpi=150)
    print(f"UMAP plot saved to {umap_fig_path}")
    
    # 转换 obs 中所有对象类型的列为字符串类型，避免 h5ad 保存时的类型错误
    for col in adata_cm.obs.columns:
        if adata_cm.obs[col].dtype == 'object':
            adata_cm.obs[col] = adata_cm.obs[col].astype(str)
    
    combined_path = output_dir / "combined.h5ad"
    adata_cm.write_h5ad(combined_path)
    print(f"Combined data saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run PAMONA method for multi-modal integration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--data-root', default='data', help='Data root directory')
    parser.add_argument('--no-prior', action='store_true', 
                        help='Disable cell type prior information (default: use prior)')
    parser.add_argument('--seed', type=int, default=24, help='Random seed (default: 24)')
    parser.add_argument('--rna-celltype-key', default='cell_type',
                        help='Key for RNA cell type annotation in AnnData.obs (default: cell_type)')
    parser.add_argument('--atac-celltype-key', default='pred',
                        help='Key for ATAC cell type annotation in AnnData.obs (default: pred)')
    
    args = parser.parse_args()
    
    # 确保使用绝对路径，避免相对路径问题
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Running PAMONA on dataset: {args.dataset}")
    print(f"Use prior: {not args.no_prior}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    try:
        set_random_seed(args.seed)
        
        adata_rna, adata_atac = load_data(args.dataset, args.data_root)
        data, datatype = prepare_data(adata_rna, adata_atac, 
                                      rna_celltype_key=args.rna_celltype_key,
                                      atac_celltype_key=args.atac_celltype_key)
        
        M = create_prior_matrix(data, datatype, use_prior=not args.no_prior)
        integrated_data = run_pamona(data, M=M)
        
        adata_rna, adata_atac = save_results(adata_rna, adata_atac, integrated_data, output_dir)
        generate_umap(adata_rna, adata_atac, output_dir)
        
        print("=" * 60)
        print("PAMONA completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
