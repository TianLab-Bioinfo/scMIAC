#!/usr/bin/env python
# MultiVI 方法运行脚本
"""
运行 MultiVI 方法进行多模态整合

MultiVI 是基于 scvi-tools 的多模态整合方法，可以联合分析配对(multiome)和非配对(单模态)的
scRNA-seq 和 scATAC-seq 数据。MultiVI 使用配对数据作为锚点来对齐和合并不同模态的潜在空间。

参考文档:
- https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/MultiVI_tutorial.html

Usage:
    python main.py --dataset 10x --output-dir data/10x/output/methods/multivi --paired-cells-file data/10x/input/paired_cells/paired_0.2_cells.txt
    python main.py --dataset share --output-dir data/share/output/methods/multivi --paired-cells-file data/share/input/paired_cells/paired_0.3_cells.txt
"""

import argparse
import sys
import warnings
from pathlib import Path

import anndata
import muon
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

# 添加父目录到路径以导入utils模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from utils.gpu_monitor import GPUMonitor, save_gpu_stats, PYNVML_AVAILABLE
except ImportError as e:
    print(f"Warning: Failed to import GPU monitor: {e}", file=sys.stderr)
    PYNVML_AVAILABLE = False
    GPUMonitor = None
    save_gpu_stats = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import scvi
except ImportError:
    print("Error: scvi-tools not installed. Please install it first:", file=sys.stderr)
    print("  pip install scvi-tools", file=sys.stderr)
    sys.exit(1)


def set_random_seed(seed: int = 42):
    """设置随机种子"""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    scvi.settings.seed = seed
    print(f"Random seed set to {seed}")


def load_data(dataset_name: str, data_root: str = "data"):
    """
    加载预处理的 RNA 和 ATAC peak 数据
    
    注意：使用 ATAC peak 数据（chr:start-end 格式），而非基因活性分数
    这与 MultiVI 官方教程保持一致
    """
    print(f"Loading {dataset_name} dataset...")
    
    data_dir = Path(data_root) / dataset_name / "input"
    
    rna_path = data_dir / f"adata_rna_{dataset_name}.h5ad"
    atac_path = data_dir / f"adata_peak_{dataset_name}.h5ad"  # 使用 peak 数据
    
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA data not found: {rna_path}")
    if not atac_path.exists():
        raise FileNotFoundError(f"ATAC peak data not found: {atac_path}")
    
    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)
    
    # 使用原始计数数据
    if adata_rna.raw is not None:
        print("  Using RNA raw data (original counts)")
        adata_rna = adata_rna.raw.to_adata()
    if adata_atac.raw is not None:
        print("  Using ATAC raw data (original counts)")
        adata_atac = adata_atac.raw.to_adata()
    
    print(f"RNA: {adata_rna.shape}, ATAC: {adata_atac.shape}")
    print(f"RNA features (genes): {adata_rna.var_names[:5].tolist()} ...")
    print(f"ATAC features (peaks): {adata_atac.var_names[:5].tolist()} ...")
    
    # 检查数据类型
    if hasattr(adata_rna.X, 'toarray'):
        rna_sample = adata_rna.X[:100].toarray()
    else:
        rna_sample = adata_rna.X[:100]
    
    if hasattr(adata_atac.X, 'toarray'):
        atac_sample = adata_atac.X[:100].toarray()
    else:
        atac_sample = adata_atac.X[:100]
    
    rna_non_zero = rna_sample[rna_sample > 0]
    atac_non_zero = atac_sample[atac_sample > 0]
    
    if len(rna_non_zero) > 0:
        rna_is_int = np.allclose(rna_non_zero, rna_non_zero.astype(int))
        print(f"RNA data type: {'Integer counts' if rna_is_int else 'Float (normalized/pseudo-counts)'}")
    
    if len(atac_non_zero) > 0:
        atac_is_int = np.allclose(atac_non_zero, atac_non_zero.astype(int))
        print(f"ATAC data type: {'Integer counts' if atac_is_int else 'Float (normalized/pseudo-counts)'}")
    
    return adata_rna, adata_atac


def create_partial_pairing(adata_rna, adata_atac, paired_cells_file: str):
    """
    创建部分配对的数据场景，通过将非配对细胞拆分为独立的 RNA-only 和 ATAC-only 细胞
    
    策略：
    - 从文件读取配对细胞列表（barcode 保持原样）
    - 剩余细胞拆分为：
      · RNA-only: barcode 加 "-RNA" 后缀
      · ATAC-only: barcode 加 "-ATAC" 后缀
    
    Args:
        adata_rna: RNA AnnData对象
        adata_atac: ATAC AnnData对象  
        paired_cells_file: 配对细胞列表文件路径（必需），使用文件中的细胞作为配对细胞
        
    Returns:
        adata_rna_new, adata_atac_new: 重组后的AnnData对象，包含配对+单模态细胞
    """
    print(f"\nCreating partial pairing scenario...")
    
    # 确保两个数据有相同的细胞
    common_cells = list(set(adata_rna.obs_names) & set(adata_atac.obs_names))
    if len(common_cells) == 0:
        raise ValueError("RNA and ATAC data have no common cells!")
    
    common_cells = sorted(common_cells)
    adata_rna = adata_rna[common_cells].copy()
    adata_atac = adata_atac[common_cells].copy()
    
    n_cells = len(common_cells)
    print(f"Total cells: {n_cells}")
    
    # 从文件读取配对细胞列表
    print(f"  Reading paired cells from: {paired_cells_file}")
    paired_cells_file_path = Path(paired_cells_file)
    if not paired_cells_file_path.exists():
        raise FileNotFoundError(f"Paired cells file not found: {paired_cells_file}")
    
    with open(paired_cells_file_path, 'r') as f:
        paired_cells = [line.strip() for line in f if line.strip()]
    
    # 验证配对细胞是否在数据中
    paired_cells_set = set(paired_cells)
    common_cells_set = set(common_cells)
    
    invalid_cells = paired_cells_set - common_cells_set
    if invalid_cells:
        raise ValueError(f"Found {len(invalid_cells)} cells in paired_cells_file that are not in the data. "
                       f"First few: {list(invalid_cells)[:5]}")
    
    # 过滤出在数据中的配对细胞（保持文件中的顺序）
    paired_cells_list = [c for c in paired_cells if c in common_cells_set]
    
    n_paired = len(paired_cells_list)
    n_unpaired = n_cells - n_paired
    actual_ratio = n_paired / n_cells
    
    print(f"  Paired cells from file: {n_paired} ({actual_ratio:.1%})")
    print(f"  Unpaired: {n_unpaired} ({(1-actual_ratio):.1%})")
    print(f"    → Will be split into {n_unpaired} RNA-only + {n_unpaired} ATAC-only")
    
    # 创建配对和非配对细胞的索引
    common_cells_dict = {cell: idx for idx, cell in enumerate(common_cells)}
    paired_idx = np.array([common_cells_dict[cell] for cell in paired_cells_list])
    unpaired_idx = np.array([i for i in range(n_cells) if common_cells[i] not in paired_cells_set])
    
    # === 创建配对细胞数据 ===
    adata_rna_paired = adata_rna[paired_idx].copy()
    adata_atac_paired = adata_atac[paired_idx].copy()
    adata_rna_paired.obs['modality_type'] = 'paired'
    adata_atac_paired.obs['modality_type'] = 'paired'
    
    # === 创建 RNA-only 细胞 ===
    adata_rna_only = adata_rna[unpaired_idx].copy()
    # 修改 barcode 加后缀
    adata_rna_only.obs_names = [f"{name}-RNA" for name in adata_rna_only.obs_names]
    adata_rna_only.obs['modality_type'] = 'rna_only'
    
    # 创建对应的零 ATAC 数据
    adata_atac_rna_only = adata_atac[unpaired_idx].copy()
    adata_atac_rna_only.obs_names = [f"{name}-RNA" for name in adata_atac_rna_only.obs_names]
    # 将 ATAC 数据置零
    if sp.issparse(adata_atac_rna_only.X):
        adata_atac_rna_only.X = sp.csr_matrix(adata_atac_rna_only.X.shape)
    else:
        adata_atac_rna_only.X = np.zeros_like(adata_atac_rna_only.X)
    adata_atac_rna_only.obs['modality_type'] = 'rna_only'
    
    # === 创建 ATAC-only 细胞 ===
    adata_atac_only = adata_atac[unpaired_idx].copy()
    # 修改 barcode 加后缀
    adata_atac_only.obs_names = [f"{name}-ATAC" for name in adata_atac_only.obs_names]
    adata_atac_only.obs['modality_type'] = 'atac_only'
    
    # 创建对应的零 RNA 数据
    adata_rna_atac_only = adata_rna[unpaired_idx].copy()
    adata_rna_atac_only.obs_names = [f"{name}-ATAC" for name in adata_rna_atac_only.obs_names]
    # 将 RNA 数据置零
    if sp.issparse(adata_rna_atac_only.X):
        adata_rna_atac_only.X = sp.csr_matrix(adata_rna_atac_only.X.shape)
    else:
        adata_rna_atac_only.X = np.zeros_like(adata_rna_atac_only.X)
    adata_rna_atac_only.obs['modality_type'] = 'atac_only'
    
    # === 合并所有细胞 ===
    # RNA 模态: 配对 + RNA-only + ATAC-only(零数据)
    adata_rna_new = anndata.concat(
        [adata_rna_paired, adata_rna_only, adata_rna_atac_only],
        join='outer',
        merge='same'
    )
    
    # ATAC 模态: 配对 + RNA-only(零数据) + ATAC-only
    adata_atac_new = anndata.concat(
        [adata_atac_paired, adata_atac_rna_only, adata_atac_only],
        join='outer',
        merge='same'
    )
    
    # 确保细胞顺序一致
    assert all(adata_rna_new.obs_names == adata_atac_new.obs_names), \
        "Cell names must match between RNA and ATAC!"
    
    print("Partial pairing created successfully")
    print(f"\nFinal data structure:")
    print(f"  Total cells: {adata_rna_new.n_obs}")
    print(f"    - Paired: {n_paired} (barcode unchanged)")
    print(f"    - RNA-only: {n_unpaired} (barcode + '-RNA')")
    print(f"    - ATAC-only: {n_unpaired} (barcode + '-ATAC')")
    print(f"\n  RNA modality: {adata_rna_new.n_obs} cells")
    print(f"    - {n_paired} paired with RNA data")
    print(f"    - {n_unpaired} RNA-only with RNA data")
    print(f"    - {n_unpaired} ATAC-only with zero RNA data")
    print(f"\n  ATAC modality: {adata_atac_new.n_obs} cells")
    print(f"    - {n_paired} paired with ATAC data")
    print(f"    - {n_unpaired} RNA-only with zero ATAC data")
    print(f"    - {n_unpaired} ATAC-only with ATAC data")
    
    return adata_rna_new, adata_atac_new


def preprocess_for_multivi(adata_rna, adata_atac, n_top_genes: int = 4000):
    """
    为 MultiVI 进行数据预处理
    
    MultiVI 要求：
    1. RNA 和 ATAC 可以有不同的特征空间（基因 vs peaks）
    2. 使用原始计数数据
    3. 选择高变基因/高变peaks
    
    注意：与官方教程一致，RNA 和 ATAC 特征空间完全不同是正常的
    MultiVI 通过潜在空间学习跨模态的对应关系
    """
    print("\nPreprocessing for MultiVI...")
    
    # 备份原始计数
    if 'counts' not in adata_rna.layers:
        adata_rna.layers['counts'] = adata_rna.X.copy()
    if 'counts' not in adata_atac.layers:
        adata_atac.layers['counts'] = adata_atac.X.copy()
    
    # 为每个模态选择高变特征
    print(f"  Selecting top {n_top_genes} features for each modality...")
    
    # RNA: 选择高变基因
    print(f"  RNA: selecting highly variable genes...")
    sc.pp.highly_variable_genes(
        adata_rna,
        n_top_genes=n_top_genes,
        flavor='seurat_v3',
        layer='counts',
        subset=True
    )
    print(f"    RNA after filtering: {adata_rna.shape}")
    print(f"    Sample features: {adata_rna.var_names[:3].tolist()} ...")
    
    # ATAC: 选择高变peaks
    print(f"  ATAC: selecting highly variable peaks...")
    sc.pp.highly_variable_genes(
        adata_atac,
        n_top_genes=n_top_genes,
        flavor='seurat_v3',
        layer='counts',
        subset=True
    )
    print(f"    ATAC after filtering: {adata_atac.shape}")
    print(f"    Sample features: {adata_atac.var_names[:3].tolist()} ...")
    
    return adata_rna, adata_atac


def create_mudata(adata_rna, adata_atac):
    """
    创建 MuData 对象
    
    MultiVI 要求使用 MuData 格式来组织多模态数据
    """
    print("\nCreating MuData object...")
    
    # 确保细胞名称一致
    if not all(adata_rna.obs_names == adata_atac.obs_names):
        common_cells = list(set(adata_rna.obs_names) & set(adata_atac.obs_names))
        common_cells = sorted(common_cells)
        adata_rna = adata_rna[common_cells].copy()
        adata_atac = adata_atac[common_cells].copy()
    
    # 创建 MuData 对象
    # 注意：modality keys 要和后续 setup_mudata 中的一致
    mdata = muon.MuData({
        'rna': adata_rna,
        'atac': adata_atac
    })
    
    print(f"MuData created: {mdata}")
    
    # 转换为 CSR 格式以加速训练
    if sp.issparse(mdata.mod['rna'].X):
        mdata.mod['rna'].X = mdata.mod['rna'].X.tocsr()
    if sp.issparse(mdata.mod['atac'].X):
        mdata.mod['atac'].X = mdata.mod['atac'].X.tocsr()
    
    mdata.update()
    
    return mdata


def train_multivi_model(mdata, n_latent: int = None, n_layers_encoder: int = 2,
                       n_layers_decoder: int = 2, max_epochs: int = 500, device: str = 'cuda:0'):
    """训练 MultiVI 模型"""
    print("\nTraining MultiVI model...")
    
    # Setup MuData for MultiVI
    # 注意：modality keys 需要和 MuData 中的一致
    scvi.model.MULTIVI.setup_mudata(
        mdata,
        modalities={
            'rna_layer': 'rna',
            'atac_layer': 'atac',
        }
    )
    
    # 创建 MultiVI 模型
    # 需要指定基因数量和区域数量
    n_genes = mdata.mod['rna'].n_vars
    n_regions = mdata.mod['atac'].n_vars
    
    print(f"  n_genes: {n_genes}")
    print(f"  n_regions: {n_regions}")
    print(f"  n_latent: {n_latent} (None = use default)")
    print(f"  n_layers_encoder: {n_layers_encoder}")
    print(f"  n_layers_decoder: {n_layers_decoder}")
    
    # 构建模型参数，只传递非 None 的参数以使用官方默认值
    model_kwargs = {
        'n_genes': n_genes,
        'n_regions': n_regions,
        'n_layers_encoder': n_layers_encoder,
        'n_layers_decoder': n_layers_decoder
    }
    if n_latent is not None:
        model_kwargs['n_latent'] = n_latent
    
    model = scvi.model.MULTIVI(mdata, **model_kwargs)
    
    # 查看模型设置
    model.view_anndata_setup()
    
    # 训练模型
    print(f"\n  Training on device: {device}")
    print(f"  Max epochs: {max_epochs}")
    
    # 设置设备参数（新版 scvi-tools API）
    train_kwargs = {'max_epochs': max_epochs}
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("  WARNING: CUDA not available, using CPU instead")
            train_kwargs['accelerator'] = 'cpu'
        else:
            gpu_id = int(device.split(':')[-1]) if ':' in device else 0
            train_kwargs['accelerator'] = 'gpu'
            train_kwargs['devices'] = [gpu_id]
    else:
        train_kwargs['accelerator'] = 'cpu'
    
    model.train(**train_kwargs)
    
    print("MultiVI training completed")
    return model


def extract_embeddings(model, mdata):
    """提取 MultiVI 潜在表示"""
    print("\nExtracting latent representations...")
    
    latent = model.get_latent_representation(mdata)
    mdata.obsm['X_multivi'] = latent
    
    print(f"Latent representation shape: {latent.shape}")
    return mdata


def save_results(mdata, output_dir: Path):
    """保存潜在表示（只保存未配对细胞，配对细胞作为bridge不保存）"""
    print("\nSaving latent representations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MultiVI 在部分配对场景下包含：配对、RNA-only、ATAC-only 三类细胞
    # 只保存未配对细胞（与 Seurat5 保持一致）：
    # - RNA embeddings: 只保存 RNA-only 细胞（带 -RNA 后缀）
    # - ATAC embeddings: 只保存 ATAC-only 细胞（带 -ATAC 后缀）
    # - 配对细胞作为 bridge，不保存在输出文件中
    
    # 检查 modality_type（可能在 mdata.obs 或 mdata.mod['rna'].obs 中）
    modality_type_col = None
    if 'modality_type' in mdata.obs.columns:
        modality_type_col = mdata.obs['modality_type']
    elif 'rna' in mdata.mod and 'modality_type' in mdata.mod['rna'].obs.columns:
        modality_type_col = mdata.mod['rna'].obs['modality_type']
    elif 'atac' in mdata.mod and 'modality_type' in mdata.mod['atac'].obs.columns:
        modality_type_col = mdata.mod['atac'].obs['modality_type']
    
    # 根据 modality_type 过滤细胞
    if modality_type_col is not None:
        # RNA embeddings: 只保存 RNA-only 细胞（不包含配对细胞）
        rna_mask = modality_type_col == 'rna_only'
        
        # ATAC embeddings: 只保存 ATAC-only 细胞（不包含配对细胞）
        atac_mask = modality_type_col == 'atac_only'
    elif 'modality' in mdata.obs.columns:
        # 旧的模态标识方式
        rna_mask = mdata.obs['modality'] == 'RNA'
        atac_mask = mdata.obs['modality'] == 'ATAC'
    else:
        # 默认：保存所有细胞
        rna_mask = np.ones(mdata.n_obs, dtype=bool)
        atac_mask = np.ones(mdata.n_obs, dtype=bool)
    
    # 保存为 CSV 格式
    rna_embeddings_csv = output_dir / "rna_embeddings.csv"
    atac_embeddings_csv = output_dir / "atac_embeddings.csv"
    
    # 创建 RNA embeddings DataFrame（保持原始顺序）
    rna_df = pd.DataFrame(
        mdata.obsm['X_multivi'][rna_mask],
        index=mdata.obs_names[rna_mask]
    )
    rna_df.to_csv(rna_embeddings_csv)
    
    # 创建 ATAC embeddings DataFrame（保持原始顺序）
    atac_df = pd.DataFrame(
        mdata.obsm['X_multivi'][atac_mask],
        index=mdata.obs_names[atac_mask]
    )
    atac_df.to_csv(atac_embeddings_csv)
    
    print(f"RNA embeddings saved to {rna_embeddings_csv} ({rna_mask.sum()} cells)")
    print(f"  Cell naming: RNA-only cells with -RNA suffix")
    print(f"ATAC embeddings saved to {atac_embeddings_csv} ({atac_mask.sum()} cells)")
    print(f"  Cell naming: ATAC-only cells with -ATAC suffix")
    print(f"  Note: Paired cells (bridge) are excluded from saved embeddings")
    
    # 打印模态类型分布信息（如果存在）
    if modality_type_col is not None:
        print(f"\n  Modality type distribution:")
        modality_counts = modality_type_col.value_counts().to_dict()
        for mod_type, count in sorted(modality_counts.items()):
            print(f"    {mod_type}: {count}")
        print(f"  Note: Modality type information is stored in combined.h5mu")


def generate_umap(mdata, output_dir: Path):
    """生成 UMAP 可视化（将 RNA 和 ATAC 分别提取并合并为 AnnData）"""
    print("\nGenerating UMAP visualization...")
    
    # 从 MuData 中提取嵌入和元数据
    latent = mdata.obsm['X_multivi']
    obs_df = mdata.obs.copy()
    
    # 检查 cell_type 列的位置
    if 'cell_type' not in obs_df.columns:
        # 尝试从 rna 模态获取
        if 'rna' in mdata.mod and 'cell_type' in mdata.mod['rna'].obs.columns:
            obs_df['cell_type'] = mdata.mod['rna'].obs['cell_type']
        elif 'atac' in mdata.mod and 'cell_type' in mdata.mod['atac'].obs.columns:
            obs_df['cell_type'] = mdata.mod['atac'].obs['cell_type']
    
    # 检查 modality_type 列
    if 'modality_type' not in obs_df.columns:
        if 'rna' in mdata.mod and 'modality_type' in mdata.mod['rna'].obs.columns:
            obs_df['modality_type'] = mdata.mod['rna'].obs['modality_type']
        elif 'atac' in mdata.mod and 'modality_type' in mdata.mod['atac'].obs.columns:
            obs_df['modality_type'] = mdata.mod['atac'].obs['modality_type']
    
    # 创建临时 AnnData 用于 UMAP 可视化
    adata_combined = anndata.AnnData(
        X=latent,
        obs=obs_df,
        obsm={'X_multivi': latent}
    )
    
    # 计算 neighbors 和 UMAP
    sc.pp.neighbors(adata_combined, use_rep='X_multivi')
    sc.tl.umap(adata_combined, min_dist=0.2)
    
    # 保存 UMAP 图片
    umap_fig_path = output_dir / "multivi_latent_umap.png"
    
    # 选择可视化的变量
    color_list = []
    if 'cell_type' in adata_combined.obs.columns:
        color_list.append('cell_type')
    if 'modality_type' in adata_combined.obs.columns:
        color_list.append('modality_type')
    
    # 如果没有任何可视化列，至少生成一个基本的 UMAP
    if not color_list:
        print("  Warning: No cell_type or modality_type found, generating basic UMAP")
        fig = sc.pl.umap(
            adata_combined,
            show=False,
            return_fig=True
        )
    else:
        fig = sc.pl.umap(
            adata_combined,
            color=color_list,
            legend_loc="on data",
            show=False,
            return_fig=True
        )
    
    fig.savefig(umap_fig_path, bbox_inches='tight', dpi=150)
    print(f"UMAP plot saved to {umap_fig_path}")
    
    # 清理保留列名（避免保存 MuData 时出错）
    for mod_name in mdata.mod.keys():
        if '_index' in mdata.mod[mod_name].var.columns:
            mdata.mod[mod_name].var.drop(columns=['_index'], inplace=True)
            print(f"  Removed reserved column '_index' from {mod_name} var")
    
    # 保存 MuData
    combined_path = output_dir / "combined.h5mu"
    mdata.write(combined_path)
    print(f"Combined MuData saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run MultiVI method for multi-modal integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset 10x --output-dir data/10x/output/methods/multivi --paired-cells-file data/10x/input/paired_cells/paired_0.2_cells.txt
  python main.py --dataset share --output-dir data/share/output/methods/multivi --paired-cells-file data/share/input/paired_cells/paired_0.3_cells.txt
  python main.py --dataset 10x --output-dir data/10x/output/methods/multivi --paired-cells-file data/10x/input/paired_cells/paired_0.1_cells.txt --device cpu
"""
    )
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--data-root', default='data', help='Data root directory')
    parser.add_argument('--paired-cells-file', type=str, required=True,
                        help='Path to file containing paired cell barcodes (one per line). Required.')
    parser.add_argument('--n-top-genes', type=int, default=4000,
                        help='Number of highly variable features per modality (default: 4000)')
    parser.add_argument('--n-latent', type=int, default=None,
                        help='Latent dimension (default: None, use MultiVI default)')
    parser.add_argument('--n-layers-encoder', type=int, default=2,
                        help='Number of encoder hidden layers (default: 2)')
    parser.add_argument('--n-layers-decoder', type=int, default=2,
                        help='Number of decoder hidden layers (default: 2)')
    parser.add_argument('--max-epochs', type=int, default=500,
                        help='Max epochs for training (default: 500, MultiVI official default)')
    parser.add_argument('--device', default='cuda:0',
                        help='Device (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--gpu-csv', type=str,
                        help='Custom path for GPU stats CSV file')
    
    args = parser.parse_args()
    
    # 确保使用绝对路径，避免相对路径问题
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Running MultiVI on dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Paired cells file: {args.paired_cells_file}")
    print(f"Device: {args.device}")
    print(f"Max epochs: {args.max_epochs} (MultiVI default: 500)")
    print(f"Latent dimension: {args.n_latent if args.n_latent else 'default'}")
    print("=" * 60)
    
    # 启动GPU监控（如果使用GPU）
    gpu_monitor = None
    gpu_id = 0
    if args.device.startswith('cuda') and PYNVML_AVAILABLE and GPUMonitor:
        try:
            gpu_id = int(args.device.split(':')[-1]) if ':' in args.device else 0
            gpu_monitor = GPUMonitor(gpu_id=gpu_id, sampling_interval=1.0)
            gpu_monitor.start()
        except Exception as e:
            print(f"Warning: Failed to start GPU monitoring: {e}")
            gpu_monitor = None
    
    try:
        set_random_seed(args.seed)
        
        # 加载数据
        adata_rna, adata_atac = load_data(args.dataset, args.data_root)
        
        # 创建部分配对场景
        adata_rna, adata_atac = create_partial_pairing(
            adata_rna, adata_atac,
            paired_cells_file=args.paired_cells_file
        )
        
        # 预处理
        adata_rna, adata_atac = preprocess_for_multivi(
            adata_rna, adata_atac,
            n_top_genes=args.n_top_genes
        )
        
        # 创建 MuData
        mdata = create_mudata(adata_rna, adata_atac)
        
        # 训练 MultiVI 模型
        model = train_multivi_model(
            mdata,
            n_latent=args.n_latent,
            n_layers_encoder=args.n_layers_encoder,
            n_layers_decoder=args.n_layers_decoder,
            max_epochs=args.max_epochs,
            device=args.device
        )
        
        # 提取嵌入
        mdata = extract_embeddings(model, mdata)
        
        # 保存结果
        save_results(mdata, output_dir)
        generate_umap(mdata, output_dir)
        
        print("\n" + "=" * 60)
        print("MultiVI completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # 停止GPU监控并保存统计数据
        if gpu_monitor and save_gpu_stats:
            try:
                stats = gpu_monitor.stop()
                csv_path = Path(args.gpu_csv) if args.gpu_csv else None
                save_gpu_stats(
                    stats,
                    output_dir.parent,
                    method_name='multivi',
                    gpu_id=gpu_id,
                    csv_path=csv_path
                )
            except Exception as e:
                print(f"Warning: Failed to save GPU stats: {e}")


if __name__ == '__main__':
    main()
