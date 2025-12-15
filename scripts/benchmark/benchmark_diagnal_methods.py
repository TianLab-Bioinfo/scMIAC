#!/usr/bin/env python
"""
基准测试方法评估脚本 (Methods Benchmark)
自动评估所有数据集上的基准测试方法
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
EXPERIMENT = 'diagnal_methods'
N_NEIGHBORS = 10
N_CORES = 10


def main(output_filename='summary.csv'):
    """主函数"""
    print("=" * 60)
    print("Methods Benchmark Evaluation")
    print("=" * 60)
    
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        # 扫描可用方法
        methods_dir = project_root / f"data/{dataset}/output/{EXPERIMENT}"
        if not methods_dir.exists():
            print(f"  ⊗ Skipped: output directory not found")
            continue
        
        # 检查哪些方法有完整的文件
        methods = []
        for method_dir in methods_dir.iterdir():
            if method_dir.is_dir():
                rna_emb = method_dir / "rna_embeddings.csv"
                atac_emb = method_dir / "atac_embeddings.csv"
                
                if check_embeddings_exist(
                    project_root / f"data/{dataset}/input/adata_rna_{dataset}.h5ad",
                    project_root / f"data/{dataset}/input/adata_atac_{dataset}.h5ad",
                    rna_emb,
                    atac_emb
                ):
                    methods.append(method_dir.name)
        
        if not methods:
            print(f"  ⊗ Skipped: no methods with embeddings found")
            continue
        
        print(f"  Found {len(methods)} methods: {', '.join(methods)}")
        
        # 评估所有方法
        all_results = {}
        for method in sorted(methods):
            print(f"  → Evaluating {method}...")
            
            try:
                # 加载数据
                rna_emb_path = methods_dir / method / "rna_embeddings.csv"
                atac_emb_path = methods_dir / method / "atac_embeddings.csv"
                
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
            save_benchmark_summary(methods_dir / output_filename, all_results)
        else:
            print(f"  ⊗ No results to save")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基准测试方法评估脚本")
    parser.add_argument(
        '-o', '--output',
        default='summary.csv',
        help='输出文件名 (默认: summary.csv)'
    )
    args = parser.parse_args()
    main(output_filename=args.output)
