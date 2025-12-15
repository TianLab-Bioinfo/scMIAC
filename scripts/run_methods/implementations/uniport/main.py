#!/usr/bin/env python
# UniPort 方法运行脚本
"""
运行 UniPort 方法进行多模态整合，支持细胞类型先验信息

Usage:
    python uniport.py --dataset 10x --output-dir data/10x/output/methods/uniport
    python uniport.py --dataset 10x --output-dir data/10x/output/methods/uniport_noct --no-prior
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import warnings

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

# NumPy 2.0 compatibility fix for uniport
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

try:
    import uniport as up
except ImportError:
    print("Error: uniport not installed. Please install it first.", file=sys.stderr)
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
    
    adata_rna = up.load_file(str(rna_path))
    adata_atac = up.load_file(str(atac_path))
    
    if adata_rna.raw is not None:
        adata_rna = adata_rna.raw.to_adata()
    if adata_atac.raw is not None:
        adata_atac = adata_atac.raw.to_adata()
    
    print(f"RNA: {adata_rna.shape}, ATAC: {adata_atac.shape}")
    return adata_rna, adata_atac


def prepare_uniport_data(adata_rna, adata_atac):
    """准备 UniPort 输入数据"""
    print("Preparing data for UniPort...")
    
    adata_atac.obs['domain_id'] = 0
    adata_atac.obs['domain_id'] = adata_atac.obs['domain_id'].astype('category')
    adata_atac.obs['source'] = 'ATAC'
    
    adata_rna.obs['domain_id'] = 1
    adata_rna.obs['domain_id'] = adata_rna.obs['domain_id'].astype('category')
    adata_rna.obs['source'] = 'RNA'
    
    adata_cm = adata_atac.concatenate(adata_rna, join='inner', batch_key='domain_id')
    
    print("Selecting highly variable genes and normalizing...")
    sc.pp.highly_variable_genes(adata_cm, n_top_genes=2000, inplace=True, subset=True)
    up.batch_scale(adata_cm)
    
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=2000, inplace=True, subset=True)
    up.batch_scale(adata_rna)
    
    sc.pp.highly_variable_genes(adata_atac, n_top_genes=2000, inplace=True, subset=True)
    up.batch_scale(adata_atac)
    
    return adata_rna, adata_atac, adata_cm


def create_prior(adata_atac, adata_rna, alpha: float = 2.0, use_prior: bool = True, 
                 device: str = 'cuda:0', rna_celltype_key: str = 'cell_type', 
                 atac_celltype_key: str = 'pred'):
    """创建先验矩阵（使用细胞类型信息）
    
    Args:
        adata_atac: ATAC AnnData对象
        adata_rna: RNA AnnData对象
        alpha: 先验矩阵的alpha参数 (default: 2.0)
        use_prior: 是否使用先验信息 (default: True)
        device: 计算设备 (default: 'cuda:0')
        rna_celltype_key: RNA细胞类型列名 (default: 'cell_type')
        atac_celltype_key: ATAC细胞类型列名 (default: 'pred')
    
    Returns:
        先验矩阵列表或None
    """
    import torch
    
    print(f"Creating prior matrix with alpha={alpha}...")
    print(f"Using RNA cell type key: '{rna_celltype_key}'")
    print(f"Using ATAC cell type key: '{atac_celltype_key}'")
    
    # 检查 RNA 细胞类型（硬依赖）
    if rna_celltype_key not in adata_rna.obs.columns:
        raise KeyError(
            f"RNA cell type column '{rna_celltype_key}' not found in adata_rna.obs. "
            f"Available columns: {list(adata_rna.obs.columns)}. "
            f"UniPort requires cell type annotations for both modalities to ensure fair comparison. "
            f"Use --rna-celltype-key to specify the correct column name."
        )
    
    # 检查 ATAC 细胞类型（硬依赖）
    if atac_celltype_key not in adata_atac.obs.columns:
        raise KeyError(
            f"ATAC cell type column '{atac_celltype_key}' not found in adata_atac.obs. "
            f"Available columns: {list(adata_atac.obs.columns)}. "
            f"UniPort requires cell type annotations for both modalities to ensure fair comparison. "
            f"Use --atac-celltype-key to specify the correct column name."
        )
    
    # 如果用户主动禁用先验，返回 None 但不报错
    if not use_prior:
        print("Prior information disabled by user (--no-prior)")
        return None
    
    atac_cell_type = adata_atac.obs[atac_celltype_key].values
    rna_cell_type = adata_rna.obs[rna_celltype_key].values
    
    prior = up.get_prior(
        atac_cell_type,
        rna_cell_type,
        alpha=alpha
    )
    
    # 将 prior 移到 GPU
    if isinstance(prior, np.ndarray):
        prior = torch.from_numpy(prior).to(device)
    elif isinstance(prior, torch.Tensor):
        prior = prior.to(device)
    
    print(f"Prior matrix created (shape: {prior.shape}, device: {prior.device})")
    return [prior]


def run_uniport(adata_atac, adata_rna, adata_cm, prior=None, 
                lambda_s: float = 1.0, device: str = 'cuda:0', outdir: str = 'output/'):
    """运行 UniPort 整合"""
    mode = "with cell type prior" if prior is not None else "without prior"
    print(f"Running UniPort integration ({mode})...")
    
    gpu_id = int(device.split(':')[-1]) if ':' in device else 0
    
    adata = up.Run(
        adatas=[adata_atac, adata_rna],
        adata_cm=adata_cm,
        lambda_s=lambda_s,
        gpu=gpu_id,
        prior=prior,
        outdir=outdir
    )
    
    adata_atac = adata[adata.obs['domain_id'] == '0']
    adata_rna = adata[adata.obs['domain_id'] == '1']
    
    print("UniPort integration completed")
    return adata_rna, adata_atac


def save_results(adata_rna, adata_atac, output_dir: Path):
    """保存潜在表示"""
    print("Saving latent representations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为 CSV 格式
    rna_embeddings_csv = output_dir / "rna_embeddings.csv"
    atac_embeddings_csv = output_dir / "atac_embeddings.csv"
    
    pd.DataFrame(
        adata_rna.obsm["latent"],
        index=adata_rna.obs_names
    ).to_csv(rna_embeddings_csv)
    pd.DataFrame(
        adata_atac.obsm["latent"],
        index=adata_atac.obs_names
    ).to_csv(atac_embeddings_csv)
    
    print(f"RNA embeddings saved to {rna_embeddings_csv}")
    print(f"ATAC embeddings saved to {atac_embeddings_csv}")


def generate_umap(adata_rna, adata_atac, output_dir: Path):
    """生成 UMAP 可视化"""
    print("Generating UMAP visualization...")
    
    adata_rna.obs['modality'] = "RNA"
    adata_atac.obs['modality'] = "ATAC"
    
    combined = anndata.concat([adata_rna, adata_atac], join='outer')
    
    sc.pp.neighbors(combined, use_rep='latent', key_added='uniport')
    sc.tl.umap(combined, neighbors_key='uniport')
    combined.obsm['latent_umap'] = combined.obsm['X_umap']
    
    # 保存 UMAP 图片
    umap_fig_path = output_dir / "uniport_latent_umap.png"
    color_list = ["cell_type", "modality"] if "cell_type" in combined.obs else ["modality"]
    fig = sc.pl.embedding(
        combined,
        basis='X_umap',
        color=color_list,
        legend_loc="on data",
        show=False,
        return_fig=True
    )
    fig.savefig(umap_fig_path, bbox_inches='tight', dpi=150)
    print(f"UMAP plot saved to {umap_fig_path}")
    
    combined_path = output_dir / "combined.h5ad"
    combined.write_h5ad(combined_path)
    print(f"Combined data saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run UniPort method for multi-modal integration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--data-root', default='data', help='Data root directory')
    parser.add_argument('--no-prior', action='store_true', 
                        help='Disable cell type prior information (default: use prior)')
    parser.add_argument('--alpha', type=float, default=2.0, 
                        help='Prior matrix alpha parameter (default: 2.0)')
    parser.add_argument('--lambda-s', type=float, default=1.0, 
                        help='Lambda_s parameter (default: 1.0)')
    parser.add_argument('--device', default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=24, help='Random seed (default: 24)')
    parser.add_argument('--rna-celltype-key', default='cell_type',
                        help='Key for RNA cell type annotation in AnnData.obs (default: cell_type)')
    parser.add_argument('--atac-celltype-key', default='pred',
                        help='Key for ATAC cell type annotation in AnnData.obs (default: pred)')
    parser.add_argument('--gpu-csv', type=str,
                        help='Custom path for GPU stats CSV file (default: data/<dataset>/output/methods/gpu.csv)')
    
    args = parser.parse_args()
    
    # 确保使用绝对路径，避免相对路径问题
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Running UniPort on dataset: {args.dataset}")
    print(f"Use prior: {not args.no_prior}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
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
        
        adata_rna, adata_atac = load_data(args.dataset, args.data_root)
        adata_rna, adata_atac, adata_cm = prepare_uniport_data(adata_rna, adata_atac)
        
        prior = create_prior(adata_atac, adata_rna, alpha=args.alpha, 
                            use_prior=not args.no_prior, device=args.device,
                            rna_celltype_key=args.rna_celltype_key,
                            atac_celltype_key=args.atac_celltype_key)
        
        adata_rna, adata_atac = run_uniport(
            adata_atac, adata_rna, adata_cm,
            prior=prior, lambda_s=args.lambda_s, device=args.device,
            outdir=str(output_dir)
        )
        
        save_results(adata_rna, adata_atac, output_dir)
        generate_umap(adata_rna, adata_atac, output_dir)
        
        print("=" * 60)
        print("UniPort completed successfully!")
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
                # 保存GPU统计数据（确保使用绝对路径）
                csv_path = Path(args.gpu_csv).resolve() if args.gpu_csv else None
                save_gpu_stats(
                    stats,
                    output_dir.parent,
                    method_name='uniport',
                    gpu_id=gpu_id,
                    csv_path=csv_path
                )
            except Exception as e:
                print(f"Warning: Failed to save GPU stats: {e}")


if __name__ == '__main__':
    main()
