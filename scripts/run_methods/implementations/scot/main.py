#!/usr/bin/env python
# SCOTv2 方法运行脚本
"""
运行 SCOTv2 方法进行多模态整合（无监督方法，不使用细胞类型信息）

Usage:
    python scotv2.py --dataset 10x --output-dir data/10x/output/methods/scotv2
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

# 导入 SCOTv2 算法（在同一目录下）
try:
    from scotv2 import SCOTv2
except ImportError as e:
    print("Error: scotv2 module not found. Please check the installation.", file=sys.stderr)
    print(f"Error details: {e}", file=sys.stderr)
    sys.exit(1)


def set_random_seed(seed: int = 24):
    """设置随机种子"""
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def prepare_data(adata_rna, adata_atac):
    """准备 SCOTv2 输入数据"""
    print("Preparing data for SCOTv2...")
    
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
    
    adata_rna.obs['modality'] = "RNA"
    adata_atac.obs['modality'] = "ATAC"
    
    return RNA_arr, ATAC_arr


def run_scotv2(RNA_arr, ATAC_arr, device='cpu'):
    """运行 SCOTv2 整合
    
    Args:
        RNA_arr: RNA 数据的表示（PCA）
        ATAC_arr: ATAC 数据的表示（LSI）
        device: 计算设备 ('cpu' 或 'cuda'，默认 'cpu')
    """
    print("Running SCOTv2 integration...")
    print(f"Using device: {device}")
    
    scot_instance = SCOTv2([RNA_arr, ATAC_arr], device=device)
    integrated_data = scot_instance.align()
    
    print("SCOTv2 integration completed")
    return integrated_data


def save_results(adata_rna, adata_atac, integrated_data, output_dir: Path):
    """保存潜在表示"""
    print("Saving latent representations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rna_embeddings = integrated_data[0]
    atac_embeddings = integrated_data[1]
    
    adata_rna.obsm['scotv2_latent'] = rna_embeddings
    adata_atac.obsm['scotv2_latent'] = atac_embeddings
    
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
    
    adata_cm = anndata.concat([adata_rna, adata_atac], join='outer')
    
    sc.pp.neighbors(adata_cm, use_rep='scotv2_latent', key_added='scotv2')
    sc.tl.umap(adata_cm, neighbors_key='scotv2')
    adata_cm.obsm['scotv2_latent_umap'] = adata_cm.obsm['X_umap']
    
    # 保存 UMAP 图片
    umap_fig_path = output_dir / "scotv2_latent_umap.png"
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
        description='Run SCOTv2 method for multi-modal integration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--data-root', default='data', help='Data root directory')
    parser.add_argument('--seed', type=int, default=24, help='Random seed (default: 24)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                        help='Device for computation: cpu or cuda (default: cpu)')
    parser.add_argument('--gpu-csv', type=str,
                        help='Custom path for GPU stats CSV file (default: data/<dataset>/output/methods/gpu.csv)')
    
    args = parser.parse_args()
    
    # 确保使用绝对路径，避免相对路径问题
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Running SCOTv2 on dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # 启动GPU监控（如果使用GPU）
    gpu_monitor = None
    gpu_id = 0
    if args.device == 'cuda' and PYNVML_AVAILABLE and GPUMonitor:
        try:
            gpu_monitor = GPUMonitor(gpu_id=gpu_id, sampling_interval=1.0)
            gpu_monitor.start()
        except Exception as e:
            print(f"Warning: Failed to start GPU monitoring: {e}")
            gpu_monitor = None
    
    try:
        set_random_seed(args.seed)
        
        adata_rna, adata_atac = load_data(args.dataset, args.data_root)
        RNA_arr, ATAC_arr = prepare_data(adata_rna, adata_atac)
        
        integrated_data = run_scotv2(RNA_arr, ATAC_arr, device=args.device)
        
        adata_rna, adata_atac = save_results(adata_rna, adata_atac, integrated_data, output_dir)
        generate_umap(adata_rna, adata_atac, output_dir)
        
        print("=" * 60)
        print("SCOTv2 completed successfully!")
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
                    method_name='scotv2',
                    gpu_id=gpu_id,
                    csv_path=csv_path
                )
            except Exception as e:
                print(f"Warning: Failed to save GPU stats: {e}")


if __name__ == '__main__':
    main()
