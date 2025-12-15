#!/usr/bin/env python
# scANVI 方法运行脚本
"""
运行 scANVI 方法进行多模态整合

scANVI (Single-cell ANnotation using Variational Inference) 是基于 scVI 的有监督整合方法，
结合细胞类型标注信息实现更精准的跨模态整合。

参考文档:
- https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/harmonization.html

Usage:
    python main.py --dataset 10x --output-dir data/10x/output/methods/scanvi
    python main.py --dataset share --output-dir data/share/output/methods/scanvi --max-epochs 20
"""

import argparse
import sys
import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
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


def set_random_seed(seed: int = 0):
    """设置随机种子"""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    scvi.settings.seed = seed
    print(f"Random seed set to {seed}")


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
    
    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)
    
    if adata_rna.raw is not None:
        adata_rna = adata_rna.raw.to_adata()
    if adata_atac.raw is not None:
        adata_atac = adata_atac.raw.to_adata()
    
    print(f"RNA: {adata_rna.shape}, ATAC: {adata_atac.shape}")
    return adata_rna, adata_atac


def prepare_scanvi_data(adata_rna, adata_atac):
    """准备 scANVI 输入数据：合并两个模态并添加批次标识"""
    print("Preparing data for scANVI...")
    
    # 添加模态标识
    adata_rna.obs['batch'] = 'RNA'
    adata_atac.obs['batch'] = 'ATAC'
    adata_rna.obs['modality'] = 'RNA'
    adata_atac.obs['modality'] = 'ATAC'
    
    # CRITICAL: 为确保公平比较，ATAC数据必须使用预测的细胞类型（pred列）而非真实标签
    # RNA数据使用真实的cell_type列，ATAC数据使用pred列
    if 'pred' not in adata_atac.obs.columns:
        raise ValueError(
            "ATAC data must have 'pred' column for cell type predictions. "
            "This ensures fair comparison by using predicted labels instead of ground truth."
        )
    
    # 备份ATAC的原始cell_type（如果存在）为cell_type_true
    if 'cell_type' in adata_atac.obs.columns:
        adata_atac.obs['cell_type_true'] = adata_atac.obs['cell_type'].copy()
        print("  Backed up ATAC ground truth labels to 'cell_type_true'")
    
    # 强制ATAC使用pred列作为cell_type
    adata_atac.obs['cell_type'] = adata_atac.obs['pred'].copy()
    print("  Using ATAC 'pred' column as cell_type for fair comparison")
    
    # 确保RNA有cell_type列
    if 'cell_type' not in adata_rna.obs.columns:
        raise ValueError("RNA data must have 'cell_type' column.")
    
    # 合并数据（使用 inner join 保留共同特征）
    adata = anndata.concat(
        [adata_rna, adata_atac],
        join='inner',
        label='batch',
        keys=['RNA', 'ATAC'],
        index_unique='-'
    )
    
    print(f"Combined data shape: {adata.shape}")
    print(f"  RNA cells: {(adata.obs['batch'] == 'RNA').sum()}")
    print(f"  ATAC cells: {(adata.obs['batch'] == 'ATAC').sum()}")
    print(f"Cell types found: {adata.obs['cell_type'].nunique()}")
    print(f"  {adata.obs['cell_type'].value_counts().to_dict()}")
    
    return adata


def preprocess_for_scvi(adata, n_top_genes: int = 2000):
    """为 scVI/scANVI 进行数据预处理"""
    print("Preprocessing for scVI/scANVI...")
    
    # 备份原始计数
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()
        print("  Backed up raw counts to layers['counts']")
    
    # 选择高变基因
    print(f"  Selecting top {n_top_genes} highly variable genes...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor='seurat_v3',
        layer='counts',
        batch_key='batch',
        subset=True
    )
    
    print(f"  After filtering: {adata.shape}")
    return adata


def train_scvi_model(adata, n_latent: int = 30, n_layers: int = 2, 
                     max_epochs: int = 200, device: str = 'cuda:0'):
    """训练 scVI 模型（无监督整合）"""
    print("Training scVI model (unsupervised integration)...")
    
    # 设置 scVI 的 AnnData
    scvi.model.SCVI.setup_anndata(
        adata,
        layer='counts',
        batch_key='batch'
    )
    
    # 创建 scVI 模型
    model = scvi.model.SCVI(
        adata,
        n_layers=n_layers,
        n_latent=n_latent,
        gene_likelihood='nb'
    )
    
    # 训练模型
    print(f"  Training on device: {device}")
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
    
    print("scVI training completed")
    return model


def train_scanvi_model(scvi_model, adata, labels_key: str = 'cell_type',
                       max_epochs: int = 20, n_samples_per_label: int = 100,
                       device: str = 'cuda:0'):
    """从 scVI 模型初始化并训练 scANVI 模型（有监督整合）"""
    print("Training scANVI model (supervised integration)...")
    
    # 从 scVI 模型初始化 scANVI
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        adata=adata,
        labels_key=labels_key,
        unlabeled_category='Unknown'
    )
    
    # 训练 scANVI
    print(f"  Training on device: {device}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Samples per label: {n_samples_per_label}")
    
    # 设置设备参数（新版 scvi-tools API）
    train_kwargs = {
        'max_epochs': max_epochs,
        'n_samples_per_label': n_samples_per_label
    }
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            gpu_id = int(device.split(':')[-1]) if ':' in device else 0
            train_kwargs['accelerator'] = 'gpu'
            train_kwargs['devices'] = [gpu_id]
        else:
            train_kwargs['accelerator'] = 'cpu'
    else:
        train_kwargs['accelerator'] = 'cpu'
    
    scanvi_model.train(**train_kwargs)
    
    print("scANVI training completed")
    return scanvi_model


def extract_embeddings(scanvi_model, adata):
    """提取 scANVI 潜在表示"""
    print("Extracting latent representations...")
    
    latent = scanvi_model.get_latent_representation(adata)
    adata.obsm['X_scANVI'] = latent
    
    print(f"Latent representation shape: {latent.shape}")
    return adata


def save_results(adata, output_dir: Path):
    """保存潜在表示（按模态分别保存）"""
    print("Saving latent representations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分离 RNA 和 ATAC
    rna_mask = adata.obs['batch'] == 'RNA'
    atac_mask = adata.obs['batch'] == 'ATAC'
    
    adata_rna = adata[rna_mask].copy()
    adata_atac = adata[atac_mask].copy()
    
    # 保存为 CSV 格式
    rna_embeddings_csv = output_dir / "rna_embeddings.csv"
    atac_embeddings_csv = output_dir / "atac_embeddings.csv"
    
    pd.DataFrame(
        adata_rna.obsm['X_scANVI'],
        index=adata_rna.obs_names
    ).to_csv(rna_embeddings_csv)
    
    pd.DataFrame(
        adata_atac.obsm['X_scANVI'],
        index=adata_atac.obs_names
    ).to_csv(atac_embeddings_csv)
    
    print(f"RNA embeddings saved to {rna_embeddings_csv}")
    print(f"ATAC embeddings saved to {atac_embeddings_csv}")


def generate_umap(adata, output_dir: Path):
    """生成 UMAP 可视化"""
    print("Generating UMAP visualization...")
    
    # 计算 neighbors 和 UMAP
    sc.pp.neighbors(adata, use_rep='X_scANVI', key_added='scanvi')
    sc.tl.umap(adata, neighbors_key='scanvi')
    adata.obsm['latent_umap'] = adata.obsm['X_umap']
    
    # 保存 UMAP 图片
    umap_fig_path = output_dir / "scanvi_latent_umap.png"
    color_list = ["cell_type", "batch"]
    
    fig = sc.pl.embedding(
        adata,
        basis='X_umap',
        color=color_list,
        legend_loc="right margin",
        show=False,
        return_fig=True
    )
    fig.savefig(umap_fig_path, bbox_inches='tight', dpi=150)
    print(f"UMAP plot saved to {umap_fig_path}")
    
    # 保存合并的 AnnData
    combined_path = output_dir / "combined.h5ad"
    adata.write_h5ad(combined_path)
    print(f"Combined data saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run scANVI method for multi-modal integration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--data-root', default='data', help='Data root directory')
    parser.add_argument('--n-top-genes', type=int, default=2000,
                        help='Number of highly variable genes (default: 2000)')
    parser.add_argument('--n-latent', type=int, default=30,
                        help='Latent dimension (default: 30)')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Number of hidden layers (default: 2)')
    parser.add_argument('--scvi-epochs', type=int, default=200,
                        help='Max epochs for scVI training (default: 200)')
    parser.add_argument('--scanvi-epochs', type=int, default=20,
                        help='Max epochs for scANVI training (default: 20)')
    parser.add_argument('--n-samples-per-label', type=int, default=100,
                        help='Samples per label for scANVI (default: 100)')
    parser.add_argument('--device', default='cuda:0',
                        help='Device (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--gpu-csv', type=str,
                        help='Custom path for GPU stats CSV file (default: data/<dataset>/output/methods/gpu.csv)')
    
    args = parser.parse_args()
    
    # 确保使用绝对路径，避免相对路径问题
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Running scANVI on dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"scVI epochs: {args.scvi_epochs}")
    print(f"scANVI epochs: {args.scanvi_epochs}")
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
        
        # 准备数据
        adata = prepare_scanvi_data(adata_rna, adata_atac)
        adata = preprocess_for_scvi(adata, n_top_genes=args.n_top_genes)
        
        # 训练 scVI 模型
        scvi_model = train_scvi_model(
            adata,
            n_latent=args.n_latent,
            n_layers=args.n_layers,
            max_epochs=args.scvi_epochs,
            device=args.device
        )
        
        # 训练 scANVI 模型
        scanvi_model = train_scanvi_model(
            scvi_model,
            adata,
            labels_key='cell_type',
            max_epochs=args.scanvi_epochs,
            n_samples_per_label=args.n_samples_per_label,
            device=args.device
        )
        
        # 提取嵌入
        adata = extract_embeddings(scanvi_model, adata)
        
        # 保存结果
        save_results(adata, output_dir)
        generate_umap(adata, output_dir)
        
        print("=" * 60)
        print("scANVI completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # 停止GPU监控并保存统计数据
        if gpu_monitor and save_gpu_stats:
            try:
                stats = gpu_monitor.stop()
                # 保存GPU统计数据
                csv_path = Path(args.gpu_csv) if args.gpu_csv else None
                save_gpu_stats(
                    stats,
                    output_dir.parent,
                    method_name='scanvi',
                    gpu_id=gpu_id,
                    csv_path=csv_path
                )
            except Exception as e:
                print(f"Warning: Failed to save GPU stats: {e}")


if __name__ == '__main__':
    main()
