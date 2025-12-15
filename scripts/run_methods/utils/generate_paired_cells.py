#!/usr/bin/env python
"""
生成配对细胞列表
用于在不同方法间保持一致的配对/非配对细胞划分

Usage:
    python generate_paired_cells.py --dataset 10x --ratio 0.2
    python generate_paired_cells.py --dataset share --ratio 0.3 --seed 123
"""

import argparse
from pathlib import Path
import numpy as np
import scanpy as sc


def load_cell_names(dataset_name: str, data_root: str = "data"):
    """
    加载RNA和ATAC数据的细胞名
    
    Returns:
        common_cells: 共同细胞的列表（排序后）
    """
    print(f"Loading cell names from {dataset_name} dataset...")
    
    data_dir = Path(data_root) / dataset_name / "input"
    
    # 使用 h5ad 格式（更通用）
    rna_path = data_dir / f"adata_rna_{dataset_name}.h5ad"
    atac_path = data_dir / f"adata_atac_{dataset_name}.h5ad"
    
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA data not found: {rna_path}")
    if not atac_path.exists():
        raise FileNotFoundError(f"ATAC data not found: {atac_path}")
    
    # 只读取obs_names，不加载整个数据
    adata_rna = sc.read_h5ad(rna_path, backed='r')
    adata_atac = sc.read_h5ad(atac_path, backed='r')
    
    rna_cells = set(adata_rna.obs_names)
    atac_cells = set(adata_atac.obs_names)
    
    print(f"  RNA cells: {len(rna_cells)}")
    print(f"  ATAC cells: {len(atac_cells)}")
    
    # 找到共同细胞
    common_cells = sorted(list(rna_cells & atac_cells))
    
    if len(common_cells) == 0:
        raise ValueError("No common cells found between RNA and ATAC data!")
    
    print(f"  Common cells: {len(common_cells)}")
    
    return common_cells


def generate_paired_cells(common_cells: list, ratio: float, seed: int = 42):
    """
    从共同细胞中随机选择配对细胞
    
    Args:
        common_cells: 共同细胞列表
        ratio: 配对细胞的比例 (0-1)
        seed: 随机种子
        
    Returns:
        paired_cells: 配对细胞的列表
    """
    n_cells = len(common_cells)
    n_paired = int(n_cells * ratio)
    
    print(f"\nGenerating paired cells (ratio: {ratio:.1%}, seed: {seed})...")
    print(f"  Total cells: {n_cells}")
    print(f"  Paired cells: {n_paired} ({ratio:.1%})")
    print(f"  Unpaired cells: {n_cells - n_paired} ({(1-ratio):.1%})")
    
    # 设置随机种子并随机选择
    np.random.seed(seed)
    indices = np.random.permutation(n_cells)
    paired_idx = indices[:n_paired]
    
    # 获取配对细胞（保持原始barcode，无后缀）
    paired_cells = [common_cells[i] for i in sorted(paired_idx)]
    
    return paired_cells


def save_paired_cells(paired_cells: list, output_path: Path):
    """保存配对细胞列表到文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for cell in paired_cells:
            f.write(f"{cell}\n")
    
    print(f"\nPaired cells saved to: {output_path}")
    print(f"  Total: {len(paired_cells)} cells")


def main():
    parser = argparse.ArgumentParser(
        description='Generate paired cell lists for consistent multi-method comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_paired_cells.py --dataset 10x --ratio 0.2
  python generate_paired_cells.py --dataset share --ratio 0.3 --seed 123
  
Output:
  data/{dataset}/input/paired_cells/paired_{ratio}_cells.txt
"""
    )
    parser.add_argument('--dataset', required=True,
                        choices=['10x', 'share'],
                        help='Dataset name (10x or share)')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='Ratio of paired cells (default: 0.2 = 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--data-root', default='data',
                        help='Data root directory (default: data)')
    
    args = parser.parse_args()
    
    # 验证ratio
    if not (0 < args.ratio <= 1):
        raise ValueError(f"ratio must be in (0, 1], got {args.ratio}")
    
    print("=" * 60)
    print(f"Generating paired cells for dataset: {args.dataset}")
    print(f"Ratio: {args.ratio:.1%}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # 加载细胞名
    common_cells = load_cell_names(args.dataset, args.data_root)
    
    # 生成配对细胞列表
    paired_cells = generate_paired_cells(common_cells, args.ratio, args.seed)
    
    # 保存到文件
    output_dir = Path(args.data_root) / args.dataset / "input" / "paired_cells"
    output_path = output_dir / f"paired_{args.ratio}_cells.txt"
    save_paired_cells(paired_cells, output_path)
    
    print("\n" + "=" * 60)
    print("Paired cells generation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
