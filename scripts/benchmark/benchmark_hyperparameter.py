#!/usr/bin/env python
"""
超参数实验评估脚本 (Hyperparameter Benchmark)
支持多数据集评估
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.utils import (
    check_embeddings_exist, 
    load_benchmark_data, 
    compute_all_metrics, 
    save_benchmark_summary
)

# ============ 配置 ============
EXPERIMENT = 'hyperparameter'
N_NEIGHBORS = 10


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='评估超参数实验的整合指标',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['brain', 'kidney', 'LungDroplet', '10x', 'share', 'wilk', 'zhu'],
        help='要评估的数据集列表，可指定多个数据集'
    )
    parser.add_argument(
        '--param-types',
        type=str,
        nargs='+',
        default=None,
        help='要评估的参数类型列表（如 anno_acc batch_size lambda），默认评估所有'
    )
    parser.add_argument(
        '--n-cores',
        type=int,
        default=20,
        help='计算指标时使用的CPU核心数'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=10,
        help='计算邻居相关指标时的邻居数量'
    )
    parser.add_argument(
        '-o', '--output',
        default='summary.csv',
        help='输出文件名 (默认: summary.csv)'
    )
    return parser.parse_args()


def evaluate_dataset(dataset, param_types_filter=None, n_neighbors=10, n_cores=20, output_filename='summary.csv'):
    """
    评估单个数据集的超参数实验
    
    Parameters:
    -----------
    dataset : str
        数据集名称
    param_types_filter : list of str, optional
        要评估的参数类型列表，默认评估所有
    n_neighbors : int
        邻居数量
    n_cores : int
        CPU核心数
    output_filename : str
        输出文件名
    """
    print("=" * 80)
    print(f"Dataset: {dataset}")
    print("=" * 80)
    
    hyper_dir = project_root / f"data/{dataset}/output/{EXPERIMENT}"
    if not hyper_dir.exists():
        print(f"\n⊗ Hyperparameter directory not found: {hyper_dir}")
        return
    
    # 扫描超参数类型目录（如 batch_size, lambda, anno_acc）
    param_types = [d for d in hyper_dir.iterdir() if d.is_dir()]
    
    # 应用参数类型过滤
    if param_types_filter:
        param_types = [d for d in param_types if d.name in param_types_filter]
    
    if not param_types:
        if param_types_filter:
            print(f"\n⊗ No matching parameter types found (filter: {param_types_filter})")
        else:
            print(f"\n⊗ No hyperparameter types found")
        return
    
    print(f"\nFound {len(param_types)} parameter type(s): {', '.join([p.name for p in param_types])}")
    
    # 对每种参数类型进行评估
    for param_dir in sorted(param_types):
        param_type = param_dir.name
        print(f"\n{'='*60}")
        print(f"Parameter Type: {param_type}")
        print(f"{'='*60}")
        
        # 扫描该参数类型下的所有参数值目录（支持 .npy 和 .csv 格式）
        value_dirs = []
        for value_dir in param_dir.iterdir():
            if value_dir.is_dir():
                # 优先检查 .csv，其次检查 .npy
                rna_emb = value_dir / "rna_embeddings.csv" if (value_dir / "rna_embeddings.csv").exists() else value_dir / "rna_embeddings.npy"
                atac_emb = value_dir / "atac_embeddings.csv" if (value_dir / "atac_embeddings.csv").exists() else value_dir / "atac_embeddings.npy"
                
                if check_embeddings_exist(
                    project_root / f"data/{dataset}/input/adata_rna_{dataset}.h5ad",
                    project_root / f"data/{dataset}/input/adata_atac_{dataset}.h5ad",
                    rna_emb,
                    atac_emb
                ):
                    value_dirs.append(value_dir)
        
        if not value_dirs:
            print(f"  ⊗ Skipped: no parameter values with embeddings found")
            continue
        
        print(f"  Found {len(value_dirs)} parameter values")
        
        # 评估所有参数值
        all_results = {}
        for value_dir in sorted(value_dirs):
            value_name = value_dir.name
            print(f"  → Evaluating {value_name}...")
            
            try:
                # 加载数据（支持 .csv 和 .npy 格式）
                rna_emb_path = value_dir / "rna_embeddings.csv" if (value_dir / "rna_embeddings.csv").exists() else value_dir / "rna_embeddings.npy"
                atac_emb_path = value_dir / "atac_embeddings.csv" if (value_dir / "atac_embeddings.csv").exists() else value_dir / "atac_embeddings.npy"
                
                adata_rna, adata_atac, rna_emb, atac_emb = load_benchmark_data(
                    dataset,
                    project_root / f"data/{dataset}/input/adata_rna_{dataset}.h5ad",
                    project_root / f"data/{dataset}/input/adata_atac_{dataset}.h5ad",
                    rna_emb_path,
                    atac_emb_path
                )
                
                # 计算指标
                results = compute_all_metrics(
                    adata_rna, adata_atac, rna_emb, atac_emb, 
                    dataset, value_name, 
                    n_neighbors=n_neighbors, n_cores=n_cores,
                    verbose=True
                )
                
                all_results[value_name] = results
                print(f"    ✓ Completed")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                continue
        
        # 保存结果（每个参数类型一个 summary.csv）
        if all_results:
            save_benchmark_summary(param_dir / output_filename, all_results)
        else:
            print(f"  ⊗ No results to save")
    
    print(f"\n{'='*80}")
    print(f"Dataset {dataset} evaluation complete!")
    print(f"{'='*80}\n")


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("Hyperparameter Benchmark Evaluation")
    print("=" * 80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Parameter types: {args.param_types if args.param_types else 'All'}")
    print(f"N neighbors: {args.n_neighbors}")
    print(f"N cores: {args.n_cores}")
    print(f"Output filename: {args.output}")
    print("=" * 80)
    
    # 评估每个数据集
    for dataset in args.datasets:
        evaluate_dataset(
            dataset=dataset,
            param_types_filter=args.param_types,
            n_neighbors=args.n_neighbors,
            n_cores=args.n_cores,
            output_filename=args.output
        )
    
    print("\n" + "=" * 80)
    print("All evaluations complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
