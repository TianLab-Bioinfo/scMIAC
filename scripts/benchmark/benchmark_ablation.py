#!/usr/bin/env python
"""
消融实验评估脚本 (Ablation Benchmark)
自动评估所有数据集上的消融实验，并包含 baseline (scmiac) 作为对比

用法：
    python benchmark_ablation.py                    # 运行所有数据集
    python benchmark_ablation.py -d brain           # 运行单个数据集
    python benchmark_ablation.py -d brain kidney    # 运行多个数据集
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
DATASETS = ['brain', 'kidney', 'LungDroplet', '10x', 'share', 'wilk', 'zhu']
EXPERIMENT = 'ablation'
N_NEIGHBORS = 10
N_CORES = 10


def main(datasets=None, output_filename='summary.csv'):
    """主函数"""
    # 如果没有指定数据集，使用所有数据集
    if datasets is None:
        datasets = DATASETS
    
    print("=" * 60)
    print("Ablation Benchmark Evaluation")
    print("=" * 60)
    print(f"Datasets to process: {', '.join(datasets)}\n")
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        # 检查 ablation 目录
        ablation_dir = project_root / f"data/{dataset}/output/{EXPERIMENT}"
        if not ablation_dir.exists():
            print(f"  ⊗ Skipped: ablation directory not found")
            continue
        
        # 扫描可用的消融实验方法（支持 .npy 和 .csv 格式）
        methods = []
        for method_dir in ablation_dir.iterdir():
            if method_dir.is_dir():
                # 优先检查 .csv，其次检查 .npy
                rna_emb = method_dir / "rna_embeddings.csv" if (method_dir / "rna_embeddings.csv").exists() else method_dir / "rna_embeddings.npy"
                atac_emb = method_dir / "atac_embeddings.csv" if (method_dir / "atac_embeddings.csv").exists() else method_dir / "atac_embeddings.npy"
                
                if check_embeddings_exist(
                    project_root / f"data/{dataset}/input/adata_rna_{dataset}.h5ad",
                    project_root / f"data/{dataset}/input/adata_atac_{dataset}.h5ad",
                    rna_emb,
                    atac_emb
                ):
                    methods.append(method_dir.name)
        
        # 检查并添加 baseline (scmiac from diagnal_methods/)
        baseline_dir = project_root / f"data/{dataset}/output/diagnal_methods/scmiac"
        baseline_rna_emb = baseline_dir / "rna_embeddings.csv" if (baseline_dir / "rna_embeddings.csv").exists() else baseline_dir / "rna_embeddings.npy"
        baseline_atac_emb = baseline_dir / "atac_embeddings.csv" if (baseline_dir / "atac_embeddings.csv").exists() else baseline_dir / "atac_embeddings.npy"
        baseline_exists = check_embeddings_exist(
            project_root / f"data/{dataset}/input/adata_rna_{dataset}.h5ad",
            project_root / f"data/{dataset}/input/adata_atac_{dataset}.h5ad",
            baseline_rna_emb,
            baseline_atac_emb
        )
        
        if not methods and not baseline_exists:
            print(f"  ⊗ Skipped: no methods with embeddings found")
            continue
        
        if baseline_exists:
            print(f"  ✓ Baseline scmiac found (from methods/)")
        
        if methods:
            print(f"  Found {len(methods)} ablation methods: {', '.join(methods)}")
        
        # 评估所有方法
        all_results = {}
        
        # 先评估 baseline
        if baseline_exists:
            print(f"  → Evaluating scmiac (baseline)...")
            try:
                adata_rna, adata_atac, rna_emb, atac_emb = load_benchmark_data(
                    dataset,
                    project_root / f"data/{dataset}/input/adata_rna_{dataset}.h5ad",
                    project_root / f"data/{dataset}/input/adata_atac_{dataset}.h5ad",
                    baseline_rna_emb,
                    baseline_atac_emb
                )
                
                results = compute_all_metrics(
                    adata_rna, adata_atac, rna_emb, atac_emb, 
                    dataset, 'scmiac', 
                    n_neighbors=N_NEIGHBORS, n_cores=N_CORES,
                    verbose=True
                )
                
                all_results['scmiac'] = results
                print(f"    ✓ Completed")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
        
        # 评估消融实验方法
        for method in sorted(methods):
            print(f"  → Evaluating {method}...")
            
            try:
                rna_emb_path = ablation_dir / method / "rna_embeddings.csv" if (ablation_dir / method / "rna_embeddings.csv").exists() else ablation_dir / method / "rna_embeddings.npy"
                atac_emb_path = ablation_dir / method / "atac_embeddings.csv" if (ablation_dir / method / "atac_embeddings.csv").exists() else ablation_dir / method / "atac_embeddings.npy"
                
                adata_rna, adata_atac, rna_emb, atac_emb = load_benchmark_data(
                    dataset,
                    project_root / f"data/{dataset}/input/adata_rna_{dataset}.h5ad",
                    project_root / f"data/{dataset}/input/adata_atac_{dataset}.h5ad",
                    rna_emb_path,
                    atac_emb_path
                )
                
                results = compute_all_metrics(
                    adata_rna, adata_atac, rna_emb, atac_emb, 
                    dataset, method, 
                    n_neighbors=N_NEIGHBORS, n_cores=N_CORES,
                    verbose=True
                )
                
                all_results[method] = results
                print(f"    ✓ Completed")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                continue
        
        # 保存结果
        if all_results:
            save_benchmark_summary(ablation_dir / output_filename, all_results)
        else:
            print(f"  ⊗ No results to save")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验评估脚本")
    parser.add_argument(
        '-d', '--datasets',
        nargs='+',
        choices=DATASETS,
        help=f'指定要运行的数据集 (可选: {", ".join(DATASETS)})'
    )
    parser.add_argument(
        '-o', '--output',
        default='summary.csv',
        help='输出文件名 (默认: summary.csv)'
    )
    
    args = parser.parse_args()
    main(datasets=args.datasets, output_filename=args.output)
