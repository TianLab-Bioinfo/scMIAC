#!/usr/bin/env python
"""
scMIAC 实验性工具函数

包含用于消融实验、超参数测试等场景的辅助函数。
这些函数不属于核心 scMIAC 方法，仅用于研究实验。
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path


def generate_random_anchors(
    adata_rna,
    adata_atac,
    count: int,
    seed: int | None = None,
    rna_celltype_key: str = "cell_type",
    atac_celltype_key: str = "pred",
):
    """
    生成随机锚点对（消融实验：w/o MNN）
    
    该函数用于消融实验，测试随机锚点相比 MNN 锚点的效果差异。
    随机采样后仍然保留细胞类型过滤，只是跳过了互近邻算法。

    Parameters:
    - adata_rna: AnnData containing RNA modality with cell type annotations.
    - adata_atac: AnnData containing ATAC modality with cell type annotations.
    - count: Desired number of candidate pairs prior to cell type filtering.
    - seed: Optional random seed for reproducibility.
    - rna_celltype_key: Key for cell type annotation in RNA AnnData.obs (default: 'cell_type').
    - atac_celltype_key: Key for cell type annotation in ATAC AnnData.obs (default: 'pred').

    Returns:
    - pandas.DataFrame containing filtered anchor pairs with matching cell types.
    """

    if count <= 0:
        raise ValueError("Random anchor count must be positive")

    total_rna = adata_rna.shape[0]
    total_atac = adata_atac.shape[0]
    if total_rna == 0 or total_atac == 0:
        raise ValueError("Both RNA and ATAC modalities must contain cells for random anchors")

    available_count = min(count, total_rna, total_atac)
    rng = np.random.default_rng(seed)

    rna_indices = rng.choice(total_rna, size=available_count, replace=False)
    atac_indices = rng.choice(total_atac, size=available_count, replace=False)

    if rna_celltype_key not in adata_rna.obs:
        raise KeyError(f"RNA AnnData must contain '{rna_celltype_key}' annotations for random anchors")
    if atac_celltype_key not in adata_atac.obs:
        raise KeyError(f"ATAC AnnData must contain '{atac_celltype_key}' annotations for random anchors")

    anchor_df = pd.DataFrame({
        "x1": rna_indices,
        "x2": atac_indices,
        "x1_ct": adata_rna.obs[rna_celltype_key].astype(str).to_numpy()[rna_indices],
        "x2_ct": adata_atac.obs[atac_celltype_key].astype(str).to_numpy()[atac_indices],
    })

    anchor_df["is_same"] = anchor_df["x1_ct"].astype(str) == anchor_df["x2_ct"].astype(str)
    filtered = anchor_df[anchor_df["is_same"]].copy()
    if filtered.empty:
        raise ValueError("Random anchor generation produced no matching cell types after filtering")

    filtered["score"] = 0.0
    filtered.reset_index(drop=True, inplace=True)
    print(
        f"Random anchors generated: requested={count}, available={available_count}, "
        f"retained={len(filtered)} after cell type filtering"
    )
    return filtered


def inject_annotation_noise(
    h5ad_path, 
    accuracy, 
    output_path=None, 
    seed=42,
    celltype_key="cell_type",
    unknown_label="unknown"
):
    """
    向 h5ad 文件注入注释噪声（超参数实验：注释准确率测试）
    
    将部分细胞类型标签随机替换为其他细胞类型标签，用于测试方法在不同注释准确率下的鲁棒性。
    
    Args:
        h5ad_path: 输入 h5ad 文件路径
        accuracy: 目标准确率（0-100），例如 80 表示 80% 准确
        output_path: 输出 h5ad 文件路径，如果为 None 则返回修改后的 adata 对象
        seed: 随机种子，保证可复现性
        celltype_key: 细胞类型列名（默认 "cell_type"）
        unknown_label: 未使用（保留参数以兼容旧代码）
    
    Returns:
        如果指定了 output_path，返回输出文件路径；否则返回修改后的 adata 对象
    """
    
    if accuracy < 0 or accuracy > 100:
        raise ValueError(f"Accuracy must be between 0 and 100, got {accuracy}")
    
    if accuracy == 100:
        print(f"  Accuracy is 100%, no noise injection needed")
        return h5ad_path if output_path is None else output_path
    
    np.random.seed(seed)
    
    print(f"  Loading data from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    
    # 保存原始 cell_type 到新列（如果还没保存）
    if f'{celltype_key}_original' not in adata.obs.columns:
        adata.obs[f'{celltype_key}_original'] = adata.obs[celltype_key].copy()
    
    # 获取所有唯一的细胞类型
    adata.obs[celltype_key] = adata.obs[celltype_key].astype(str)
    all_celltypes = adata.obs[celltype_key].unique()
    
    if len(all_celltypes) < 2:
        raise ValueError(f"Need at least 2 cell types to inject noise, found {len(all_celltypes)}")
    
    # 计算需要修改的细胞数量
    n_cells = adata.n_obs
    n_to_corrupt = int(n_cells * (1 - accuracy / 100))
    
    # 随机选择要修改的细胞索引
    corrupt_indices = np.random.choice(n_cells, size=n_to_corrupt, replace=False)
    
    # 对每个选中的细胞，随机替换为其他细胞类型
    n_changed = 0
    for idx in corrupt_indices:
        cell_barcode = adata.obs.index[idx]
        original_type = adata.obs.loc[cell_barcode, celltype_key]
        
        # 获取除当前类型外的其他类型
        other_types = [ct for ct in all_celltypes if ct != original_type]
        
        if len(other_types) > 0:
            # 随机选择一个不同的类型
            new_type = np.random.choice(other_types)
            adata.obs.loc[cell_barcode, celltype_key] = new_type
            n_changed += 1
    
    # 统计信息
    actual_accuracy = (1 - n_changed / n_cells) * 100
    
    print(f"  Total cells: {n_cells}")
    print(f"  Corrupted cells: {n_changed} ({n_changed/n_cells*100:.1f}%) -> random other types")
    print(f"  Actual accuracy: {actual_accuracy:.1f}%")
    
    if output_path:
        adata.write_h5ad(output_path)
        print(f"  Saved to: {output_path}")
        return output_path
    else:
        return adata


def augment_anchors_with_paired_cells(
    anchor_df: pd.DataFrame,
    paired_cells_file: str,
    adata_rna,
    adata_atac,
    rna_celltype_key: str = 'cell_type',
    atac_celltype_key: str = 'pred'
) -> pd.DataFrame:
    """
    使用配对细胞扩充anchor_df
    
    Args:
        anchor_df: 原始anchor DataFrame
        paired_cells_file: 配对细胞列表文件路径
        adata_rna: RNA AnnData对象
        adata_atac: ATAC AnnData对象
        rna_celltype_key: RNA细胞类型列名
        atac_celltype_key: ATAC细胞类型列名
        
    Returns:
        扩充后的anchor DataFrame
    """
    print(f"\nAugmenting anchors with paired cells from: {paired_cells_file}")
    
    # 读取配对细胞列表
    with open(paired_cells_file, 'r') as f:
        paired_cells = [line.strip() for line in f if line.strip()]
    
    print(f"  Read {len(paired_cells)} paired cells from file")
    
    # 找到在两个数据集中都存在的配对细胞
    rna_cells_set = set(adata_rna.obs_names)
    atac_cells_set = set(adata_atac.obs_names)
    
    valid_paired_cells = [c for c in paired_cells if c in rna_cells_set and c in atac_cells_set]
    
    print(f"  Found {len(valid_paired_cells)} valid paired cells in both datasets")
    
    if len(valid_paired_cells) == 0:
        print("  Warning: No valid paired cells found, returning original anchors")
        return anchor_df
    
    # 创建paired cells的anchor记录
    rna_cell_to_idx = {cell: idx for idx, cell in enumerate(adata_rna.obs_names)}
    atac_cell_to_idx = {cell: idx for idx, cell in enumerate(adata_atac.obs_names)}
    
    paired_anchors = []
    for cell in valid_paired_cells:
        rna_idx = rna_cell_to_idx[cell]
        atac_idx = atac_cell_to_idx[cell]
        
        rna_ct = str(adata_rna.obs[rna_celltype_key].iloc[rna_idx])
        atac_ct = str(adata_atac.obs[atac_celltype_key].iloc[atac_idx])
        
        paired_anchors.append({
            'x1': rna_idx,
            'x2': atac_idx,
            'x1_ct': rna_ct,
            'x2_ct': atac_ct,
            'is_same': (rna_ct == atac_ct),
            'source': 'paired'
        })
    
    paired_df = pd.DataFrame(paired_anchors)
    
    # 标记原始anchors的来源
    if 'source' not in anchor_df.columns:
        anchor_df = anchor_df.copy()
        anchor_df['source'] = 'mnn_or_random'
    
    # 直接合并，不去重（重复的锚点说明它们很重要，在训练时会有稍高的采样概率）
    augmented_df = pd.concat([anchor_df, paired_df], ignore_index=True)
    
    print(f"  Original anchors: {len(anchor_df)}")
    print(f"  Added paired anchors: {len(paired_df)}")
    print(f"  Total anchors: {len(augmented_df)}")
    
    # 统计is_same的比例
    n_same_original = anchor_df['is_same'].sum()
    n_same_paired = paired_df['is_same'].sum() if len(paired_df) > 0 else 0
    n_same_total = augmented_df['is_same'].sum()
    
    print(f"  Same cell type ratio:")
    print(f"    Original: {n_same_original}/{len(anchor_df)} ({n_same_original/len(anchor_df):.2%})")
    if len(paired_df) > 0:
        print(f"    Paired: {n_same_paired}/{len(paired_df)} ({n_same_paired/len(paired_df):.2%})")
    print(f"    Total: {n_same_total}/{len(augmented_df)} ({n_same_total/len(augmented_df):.2%})")
    
    return augmented_df


def save_hyperparameter_to_csv(
    csv_path,
    identifier,
    param_name,
    param_value,
    additional_info=None
):
    """
    记录超参数实验结果到 CSV 文件
    
    Args:
        csv_path: CSV 文件路径
        identifier: 实验标识符（如 "lambda_contra"）
        param_name: 参数名称
        param_value: 参数值
        additional_info: 额外信息字典
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备记录
    record = {
        'experiment': identifier,
        'parameter': param_name,
        'value': param_value,
    }
    
    if additional_info:
        record.update(additional_info)
    
    # 读取或创建 CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # 移除相同实验的旧记录
        df = df[df['experiment'] != identifier]
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    
    df.to_csv(csv_path, index=False)
    print(f"Hyperparameter info saved to: {csv_path}")
