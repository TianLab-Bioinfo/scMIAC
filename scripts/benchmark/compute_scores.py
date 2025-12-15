#!/usr/bin/env python
"""
综合评分计算脚本
从 summary.csv 中计算 BC (Biological Conservation)、MM (Modality Mixing) 和 OIS (Overall Integration Score)

用法：python compute_scores.py（自动处理所有数据集和实验，目录不存在则跳过）
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ============ 配置 ============
DATASETS = {
    'paired': ['10x', 'share'],
    'unpaired': ['brain', 'kidney', 'LungDroplet', 'wilk', 'zhu']
}

EXPERIMENTS = ['diagnal_methods', 'vertical_methods', 'ablation', 'hyperparameter']

# BC Score 指标（生物学保守性）
BC_METRICS = ['nc', 'ct_asw', 'ilasw', 'graph_connectivity', 'rct_asw']

# MM Score 指标（模态混合）- 配对数据集
MM_METRICS_PAIRED = ['batch_asw', 'ilisi', 'cilisi', 'foscttm']

# MM Score 指标（模态混合）- 未配对数据集（无 FOSCTTM）
MM_METRICS_UNPAIRED = ['batch_asw', 'ilisi', 'cilisi']

# 反向归一化的指标（越小越好）
REVERSE_METRICS = ['foscttm']

# OIS 权重
BC_WEIGHT = 0.6
MM_WEIGHT = 0.4


def get_dataset_type(dataset: str) -> str:
    """获取数据集类型"""
    if dataset in DATASETS['paired']:
        return 'paired'
    elif dataset in DATASETS['unpaired']:
        return 'unpaired'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def normalize_metrics(df: pd.DataFrame, metrics: list, reverse_metrics: list = None) -> pd.DataFrame:
    """
    归一化指标到 [0, 1] 区间
    
    Parameters:
    -----------
    df : DataFrame
        包含指标的数据框
    metrics : list
        要归一化的指标列表
    reverse_metrics : list
        需要反向归一化的指标（越小越好）
        
    Returns:
    --------
    DataFrame
        归一化后的数据框
    """
    if reverse_metrics is None:
        reverse_metrics = []
    
    df_norm = df.copy()
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        values = df[metric].dropna()
        if len(values) == 0:
            continue
            
        min_val = values.min()
        max_val = values.max()
        
        if max_val - min_val < 1e-10:  # 避免除零
            df_norm[metric] = 0.5
        else:
            if metric in reverse_metrics:
                # 反向归一化：越小的原始值 -> 越大的归一化值
                df_norm[metric] = 1 - (df[metric] - min_val) / (max_val - min_val)
            else:
                # 正向归一化：越大的原始值 -> 越大的归一化值
                df_norm[metric] = (df[metric] - min_val) / (max_val - min_val)
    
    return df_norm


def compute_bc_score(df: pd.DataFrame) -> pd.Series:
    """
    计算 BC (Biological Conservation) 得分
    BC = mean(nc, ct_asw, ilasw, graph_connectivity)
    其中 nc = mean(rna_nc, atac_nc)
    ilasw 仅在存在孤立细胞类型的数据集中计算
    """
    available_metrics = [m for m in BC_METRICS if m in df.columns]
    
    if not available_metrics:
        return pd.Series(index=df.index, dtype=float)
    
    return df[available_metrics].mean(axis=1)


def compute_mm_score(df: pd.DataFrame, dataset: str) -> pd.Series:
    """
    计算 MM (Modality Mixing) 得分
    
    配对数据集: MM = mean(batch_asw, ilisi, cilisi, foscttm)
    其中 foscttm = mean(foscttm_rna, foscttm_atac)
    未配对数据集: MM = mean(batch_asw, ilisi, cilisi)
    """
    dataset_type = get_dataset_type(dataset)
    
    if dataset_type == 'paired':
        mm_metrics = MM_METRICS_PAIRED
    else:
        mm_metrics = MM_METRICS_UNPAIRED
    
    available_metrics = [m for m in mm_metrics if m in df.columns]
    
    if not available_metrics:
        return pd.Series(index=df.index, dtype=float)
    
    return df[available_metrics].mean(axis=1)


def compute_ois(bc_score: pd.Series, mm_score: pd.Series) -> pd.Series:
    """
    计算 OIS (Overall Integration Score)
    OIS = 0.6 * BC + 0.4 * MM
    """
    return BC_WEIGHT * bc_score + MM_WEIGHT * mm_score


def find_summary_files(base_dir: Path, input_filename: str = 'summary.csv') -> list:
    """
    递归查找所有 summary.csv 文件
    
    Parameters:
    -----------
    base_dir : Path
        基础目录
    input_filename : str
        要查找的文件名
        
    Returns:
    --------
    list of Path
        找到的所有 summary.csv 文件路径
    """
    summary_files = []
    
    # 使用 rglob 递归查找所有 summary.csv
    for summary_path in base_dir.rglob(input_filename):
        summary_files.append(summary_path)
    
    return summary_files


def find_vertical_summary_files(base_dir: Path, input_prefix: str = 'summary') -> list:
    """
    查找 vertical_methods 的所有 summary_{ratio}.csv 文件
    
    Parameters:
    -----------
    base_dir : Path
        基础目录
    input_prefix : str
        输入文件名前缀
        
    Returns:
    --------
    list of Path
        找到的所有 summary_{ratio}.csv 文件路径
    """
    import re
    summary_files = []
    
    # 查找 {prefix}_*.csv 文件（例如 summary_0.2.csv, summary_0.5.csv）
    pattern = re.compile(rf'{re.escape(input_prefix)}_[\d.]+\.csv$')
    for summary_path in base_dir.glob(f"{input_prefix}_*.csv"):
        # 检查文件名是否匹配 {prefix}_{ratio}.csv 格式
        if pattern.match(summary_path.name):
            summary_files.append(summary_path)
    
    return sorted(summary_files)


def compute_scores_for_vertical_methods(dataset: str, experiment: str, input_prefix: str = 'summary', output_filename: str = 'scores.csv'):
    """
    为 vertical_methods 实验计算综合评分
    将不同配对比例的结果合并到一个 scores.csv 中
    """
    print(f"\nProcessing {dataset}/{experiment}...")
    
    experiment_dir = project_root / f"data/{dataset}/output/{experiment}"
    
    if not experiment_dir.exists():
        print(f"  ⊗ Skipped: experiment directory not found")
        return
    
    # 查找所有 summary_{ratio}.csv 文件
    summary_files = find_vertical_summary_files(experiment_dir, input_prefix)
    
    if not summary_files:
        print(f"  ⊗ Skipped: no {input_prefix}_*.csv found")
        return
    
    print(f"  Found {len(summary_files)} summary file(s)")
    
    # 收集所有配对比例的数据
    all_data = []
    
    for summary_path in summary_files:
        # 提取配对比例（从文件名 summary_0.2.csv 中提取 0.2）
        import re
        match = re.search(r'summary_([\d.]+)\.csv$', summary_path.name)
        if not match:
            continue
        ratio = match.group(1)
        
        print(f"\n  Processing: {summary_path.name}")
        
        try:
            # 读取 summary.csv
            df = pd.read_csv(summary_path, index_col=0)
            
            # 检查是否为空文件
            if df.empty or len(df.columns) == 0:
                print(f"    ⊗ Skipped: empty file")
                continue
            
            print(f"    Found {len(df)} methods")
            
            # 修改行名：method -> method({ratio} as paired cells)
            df.index = [f"{method}({ratio} as paired cells)" for method in df.index]
            
            all_data.append(df)
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue
    
    if not all_data:
        print(f"  ⊗ No valid data found")
        return
    
    # 合并所有数据
    print(f"\n  Merging all paired ratios...")
    combined_df = pd.concat(all_data, axis=0)
    
    # 归一化所有指标
    all_metrics = combined_df.columns.tolist()
    df_norm = normalize_metrics(combined_df, all_metrics, reverse_metrics=REVERSE_METRICS)
    
    # 计算 BC, MM, OIS
    bc_score = compute_bc_score(df_norm)
    mm_score = compute_mm_score(df_norm, dataset)
    ois = compute_ois(bc_score, mm_score)
    
    # 组合结果
    scores_df = pd.DataFrame({
        'BC': bc_score,
        'MM': mm_score,
        'OIS': ois
    })
    
    # 保存结果
    output_path = experiment_dir / output_filename
    scores_df.to_csv(output_path, float_format='%.5f')
    
    print(f"  ✓ Scores saved to {output_path.relative_to(experiment_dir.parent.parent)}")
    print(f"\n{scores_df}")


def compute_scores_for_experiment(dataset: str, experiment: str, input_filename: str = 'summary.csv', output_filename: str = 'scores.csv'):
    """
    为单个实验计算综合评分
    支持递归查找 summary.csv 文件（例如超参数实验的子目录）
    """
    print(f"\nProcessing {dataset}/{experiment}...")
    
    experiment_dir = project_root / f"data/{dataset}/output/{experiment}"
    
    if not experiment_dir.exists():
        print(f"  ⊗ Skipped: experiment directory not found")
        return
    
    # 递归查找所有 summary.csv 文件
    summary_files = find_summary_files(experiment_dir, input_filename)
    
    if not summary_files:
        print(f"  ⊗ Skipped: no {input_filename} found")
        return
    
    print(f"  Found {len(summary_files)} summary file(s)")
    
    # 处理每个 summary.csv
    for summary_path in sorted(summary_files):
        # 获取相对路径用于显示
        rel_path = summary_path.relative_to(experiment_dir)
        print(f"\n  Processing: {rel_path}")
        
        try:
            # 读取 summary.csv
            df = pd.read_csv(summary_path, index_col=0)
            
            # 检查是否为空文件
            if df.empty or len(df.columns) == 0:
                print(f"    ⊗ Skipped: empty file")
                continue
            
            # 按 index 排序（尝试按数字排序，否则按字符串排序）
            try:
                df['_sort_key'] = pd.to_numeric(df.index, errors='coerce')
                if not df['_sort_key'].isna().all():
                    df = df.sort_values('_sort_key').drop(columns=['_sort_key'])
                else:
                    df = df.sort_index()
            except:
                df = df.sort_index()
            
            print(f"    Found {len(df)} methods")
            
            # 归一化所有指标
            all_metrics = df.columns.tolist()
            df_norm = normalize_metrics(df, all_metrics, reverse_metrics=REVERSE_METRICS)
            
            # 计算 BC, MM, OIS
            bc_score = compute_bc_score(df_norm)
            mm_score = compute_mm_score(df_norm, dataset)
            ois = compute_ois(bc_score, mm_score)
            
            # 组合结果
            scores_df = pd.DataFrame({
                'BC': bc_score,
                'MM': mm_score,
                'OIS': ois
            })
            
            # 按 index 排序（尝试按数字排序，否则按字符串排序）
            try:
                # 尝试将 index 转换为数字并排序
                scores_df['_sort_key'] = pd.to_numeric(scores_df.index, errors='coerce')
                if not scores_df['_sort_key'].isna().all():
                    # 如果至少有一些可以转换为数字，按数字排序
                    scores_df = scores_df.sort_values('_sort_key')
                else:
                    # 否则按字符串排序
                    scores_df = scores_df.sort_index()
                scores_df = scores_df.drop(columns=['_sort_key'])
            except:
                # 如果出错，按字符串排序
                scores_df = scores_df.sort_index()
            
            # 保存结果到与 summary.csv 同级目录
            output_path = summary_path.parent / output_filename
            scores_df.to_csv(output_path, float_format='%.5f')
            
            print(f"    ✓ Scores saved to {output_path.relative_to(experiment_dir)}")
            print(f"\n{scores_df}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue


def main(input_filename='summary.csv', output_filename='scores.csv'):
    """主函数：自动处理所有数据集和实验"""
    datasets = DATASETS['paired'] + DATASETS['unpaired']
    experiments = EXPERIMENTS
    
    # 从输入文件名提取前缀（用于 vertical_methods）
    input_prefix = input_filename.rsplit('.', 1)[0]  # 'summary.csv' -> 'summary'
    
    print("=" * 60)
    print("Computing Comprehensive Scores")
    print(f"Input: {input_filename}, Output: {output_filename}")
    print("=" * 60)
    
    # 处理每个组合
    for dataset in datasets:
        for experiment in experiments:
            try:
                # vertical_methods 需要特殊处理
                if experiment == 'vertical_methods':
                    compute_scores_for_vertical_methods(dataset, experiment, input_prefix, output_filename)
                else:
                    compute_scores_for_experiment(dataset, experiment, input_filename, output_filename)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
    
    print("\n" + "=" * 60)
    print("Score computation complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="综合评分计算脚本")
    parser.add_argument(
        '-i', '--input',
        default='summary.csv',
        help='输入文件名 (默认: summary.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        default='scores.csv',
        help='输出文件名 (默认: scores.csv)'
    )
    args = parser.parse_args()
    main(input_filename=args.input, output_filename=args.output)
