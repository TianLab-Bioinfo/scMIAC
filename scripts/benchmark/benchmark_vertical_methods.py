#!/usr/bin/env python
"""
垂直整合方法评估脚本 (Vertical Integration Benchmark)
评估 vertical_methods 目录下所有方法的性能
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.utils import (
    check_embeddings_exist, 
    load_benchmark_data_vertical,
    compute_all_metrics, 
    save_benchmark_summary
)

# ============ 配置 ============
DATASETS = ['10x', 'share']
EXPERIMENT = 'vertical_methods'
PAIRED_RATIOS = ['0.2', '0.5', '0.8']
N_NEIGHBORS = 10
N_CORES = 10


def main(output_prefix='summary'):
    """主函数"""
    print("=" * 60)
    print("Vertical Integration Methods Benchmark")
    print("=" * 60)
    
    for dataset in DATASETS:
        for ratio in PAIRED_RATIOS:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset}, Paired Ratio: {ratio}")
            print(f"{'='*60}")
            
            # 扫描可用方法
            methods_dir = project_root / f"data/{dataset}/output/{EXPERIMENT}"
            if not methods_dir.exists():
                print(f"  ⊗ Skipped: output directory not found")
                continue
            
            # 检查哪些方法有对应 ratio 的结果
            methods = []
            for method_dir in methods_dir.iterdir():
                if method_dir.is_dir():
                    ratio_dir = method_dir / ratio
                    if not ratio_dir.exists():
                        continue
                    
                    rna_emb = ratio_dir / "rna_embeddings.csv"
                    atac_emb = ratio_dir / "atac_embeddings.csv"
                    
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
            
            # 读取配对细胞列表
            paired_cells_file = project_root / f"data/{dataset}/input/paired_cells/paired_{ratio}_cells.txt"
            if not paired_cells_file.exists():
                print(f"  ⊗ Skipped: paired cells file not found: {paired_cells_file}")
                continue
            
            with open(paired_cells_file, 'r') as f:
                paired_cells = set(line.strip() for line in f if line.strip())
            
            print(f"  Paired cells: {len(paired_cells)}")
            
            # 评估所有方法
            all_results = {}
            for method in sorted(methods):
                print(f"  → Evaluating {method}...")
                
                try:
                    # 加载数据
                    ratio_dir = methods_dir / method / ratio
                    rna_emb_path = ratio_dir / "rna_embeddings.csv"
                    atac_emb_path = ratio_dir / "atac_embeddings.csv"
                    
                    adata_rna, adata_atac, rna_emb, atac_emb = load_benchmark_data_vertical(
                        dataset,
                        project_root / f"data/{dataset}/input/adata_rna_{dataset}.h5ad",
                        project_root / f"data/{dataset}/input/adata_atac_{dataset}.h5ad",
                        rna_emb_path,
                        atac_emb_path,
                        paired_cells
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
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 保存结果
            if all_results:
                summary_path = methods_dir / f"{output_prefix}_{ratio}.csv"
                save_benchmark_summary(summary_path, all_results)
                print(f"  Results saved to: {summary_path}")
            else:
                print(f"  ⊗ No results to save")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="垂直整合方法评估脚本")
    parser.add_argument(
        '-o', '--output',
        default='summary',
        help='输出文件名前缀 (默认: summary，实际输出为 summary_{ratio}.csv)'
    )
    args = parser.parse_args()
    main(output_prefix=args.output)
