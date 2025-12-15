#!/bin/bash
# scMIAC 方法批量运行脚本 - 垂直整合测评（非配对数据实验）
# 在配对数据集上测试不同配对比例下，scMIAC仅使用未配对细胞的性能
# 
# 实验设计：
#   - 从数据集中排除指定比例的配对细胞（作为bridge数据）
#   - scMIAC仅在剩余的未配对细胞上运行（无先验配对信息）
#   - 测试scMIAC在纯非配对数据上的整合能力
#
# Usage:
#   bash run_scmiac.sh                    # 运行所有配对数据集和所有配对比例
#   bash run_scmiac.sh 10x                # 运行指定数据集的所有配对比例
#   bash run_scmiac.sh --help             # 显示帮助

set -e

# 设置时区为中国标准时间
export TZ='Asia/Shanghai'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMPL_SCRIPT="${SCRIPT_DIR}/../implementations/scmiac/main.py"

# 配对数据集（仅这两个支持垂直整合）
PAIRED_DATASETS=("10x" "share")

# 配对比例列表
PAIRED_RATIOS=("0.2" "0.5" "0.8")

# 显示帮助信息
show_help() {
    cat << EOF
scMIAC 方法批量运行脚本 - 垂直整合测评（非配对数据实验）

实验设计：
  本脚本测试scMIAC在纯非配对数据上的性能，配对细胞被排除而非利用。
  - 生成配对细胞列表文件（指定比例）
  - scMIAC从数据集中排除这些配对细胞
  - 仅在剩余的未配对细胞上运行MNN锚点生成和对比学习训练
  - 目标：测试scMIAC在无任何配对信息时的整合能力

Usage:
  bash run_scmiac.sh [datasets...] [extra_args...]
  
  datasets     指定要运行的配对数据集（可选，默认运行所有配对数据集）
  extra_args   传递给 scmiac main.py 的额外参数（如 --num-epochs, --device 等）

Available datasets: ${PAIRED_DATASETS[*]}
Paired ratios: ${PAIRED_RATIOS[*]}
  注：ratio表示被排除的配对细胞比例，例如0.2表示排除20%配对细胞，在80%未配对细胞上运行

Examples:
  bash run_scmiac.sh                                  # 运行所有配对数据集和所有配对比例
  bash run_scmiac.sh 10x                              # 运行 10x 数据集的所有配对比例
  bash run_scmiac.sh 10x share                        # 运行 10x 和 share 的所有配对比例
  bash run_scmiac.sh --num-epochs 50 --print-step 5   # 所有数据集+比例，自定义参数
  bash run_scmiac.sh 10x --device cuda:1              # 10x 数据集，自定义设备

Configuration:
  All parameters are passed to main.py which handles default values
  Command-line arguments can override defaults

Output:
  Results will be saved to data/<dataset>/output/vertical_methods/scmiac/<ratio>/

EOF
}

# 处理参数
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# 分离数据集和额外参数
DATASETS=()
EXTRA_ARGS=()

for arg in "$@"; do
    # 如果参数以 - 开头，认为是额外参数
    if [[ "$arg" == -* ]]; then
        EXTRA_ARGS+=("$arg")
    else
        # 检查是否是已知的配对数据集名称
        if [[ " ${PAIRED_DATASETS[*]} " =~ " ${arg} " ]]; then
            DATASETS+=("$arg")
        else
            # 不是数据集名称，当作额外参数
            EXTRA_ARGS+=("$arg")
        fi
    fi
done

# 如果没有指定数据集，使用所有配对数据集
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("${PAIRED_DATASETS[@]}")
fi

# 激活 conda 环境
echo "Activating scMIAC conda environment..."
source /usr/local/opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/nfs_share/tlj/envs/scMIAC/

echo "======================================"
echo "Running scMIAC (Unpaired Data Only)"
echo "Experiment: Exclude paired cells, run on unpaired cells only"
echo "Datasets: ${DATASETS[*]}"
echo "Paired ratios: ${PAIRED_RATIOS[*]}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra arguments: ${EXTRA_ARGS[*]}"
fi
echo "======================================"

# 记录总开始时间
TOTAL_START=$(date +%s)

# 外层循环：数据集
for dataset in "${DATASETS[@]}"; do
    # 内层循环：配对比例
    for ratio in "${PAIRED_RATIOS[@]}"; do
        output_dir="${PROJECT_ROOT}/data/${dataset}/output/vertical_methods/scmiac/${ratio}"
        paired_cells_file="${PROJECT_ROOT}/data/${dataset}/input/paired_cells/paired_${ratio}_cells.txt"
        
        echo ""
        echo "=========================================="
        echo ">>> Dataset: $dataset"
        echo ">>> Paired ratio: $ratio (cells to EXCLUDE)"
        echo ">>> Output: $output_dir"
        echo ">>> Paired cells file: $paired_cells_file"
        echo "=========================================="
        
        # 生成配对细胞文件（用于排除，不计入运行时间）
        echo ">>> Generating paired cells file for exclusion (ratio=${ratio})..."
        python "${SCRIPT_DIR}/../utils/generate_paired_cells.py" \
            --dataset "$dataset" \
            --ratio "$ratio" \
            --data-root "${PROJECT_ROOT}/data"
        
        if [ ! -f "$paired_cells_file" ]; then
            echo ">>> ERROR: Failed to generate paired cells file: $paired_cells_file" >&2
            exit 1
        fi
        
        # 记录数据集+比例的开始时间（生成配对细胞后才开始计时）
        echo ">>> Start time: $(date)"
        RUN_START=$(date +%s)
        
        # 使用 /usr/bin/time 监控内存使用
        TIME_LOG=$(mktemp)
        EXIT_CODE=0
        
        # GPU统计文件保存路径
        GPU_CSV_PATH="${PROJECT_ROOT}/data/${dataset}/output/vertical_methods/gpu_${ratio}.csv"
        
        /usr/bin/time -v python "$IMPL_SCRIPT" \
            --dataset "${dataset}" \
            --data-root "${PROJECT_ROOT}/data" \
            --output-dir "${output_dir}" \
            --paired-cells-file "${paired_cells_file}" \
            --gpu-csv "${GPU_CSV_PATH}" \
            "${EXTRA_ARGS[@]}" \
            2> "$TIME_LOG" || EXIT_CODE=$?
        
        # 检查执行是否成功
        if [ $EXIT_CODE -ne 0 ]; then
            echo ">>> ERROR: Failed to run scmiac on dataset=$dataset, ratio=$ratio (exit code: $EXIT_CODE)" >&2
            echo ">>> See error details above" >&2
            cat "$TIME_LOG" >&2
            rm -f "$TIME_LOG"
            exit $EXIT_CODE
        fi
        
        # 计算耗时
        RUN_END=$(date +%s)
        RUN_ELAPSED=$((RUN_END - RUN_START))
        RUN_MINUTES=$((RUN_ELAPSED / 60))
        RUN_SECONDS=$((RUN_ELAPSED % 60))
        
        # 提取峰值内存
        PEAK_MEMORY_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
        if [ -z "$PEAK_MEMORY_KB" ]; then
            PEAK_MEMORY_KB=0
        fi
        PEAK_MEMORY_MB=$(echo "scale=2; $PEAK_MEMORY_KB / 1024" | bc)
        PEAK_MEMORY_GB=$(echo "scale=3; $PEAK_MEMORY_KB / 1024 / 1024" | bc)
        rm -f "$TIME_LOG"
        
        echo ">>> Completed: dataset=$dataset, ratio=$ratio ($(date))"
        echo ">>> Elapsed time: ${RUN_MINUTES}m ${RUN_SECONDS}s"
        echo ">>> Peak memory: ${PEAK_MEMORY_MB} MB (${PEAK_MEMORY_GB} GB)"
        
        # 保存时间信息到 CSV
        time_dir="${PROJECT_ROOT}/data/${dataset}/output/vertical_methods"
        time_file="${time_dir}/time.csv"
        mkdir -p "$time_dir"
        
        # 如果文件不存在，创建表头
        if [ ! -f "$time_file" ]; then
            echo "method,ratio,elapsed_seconds,elapsed_time,timestamp" > "$time_file"
        fi
        
        # 覆盖已有的 scmiac-ratio 记录（如果存在）
        if grep -q "^scmiac,${ratio}," "$time_file" 2>/dev/null; then
            grep -v "^scmiac,${ratio}," "$time_file" > "${time_file}.tmp"
            mv "${time_file}.tmp" "$time_file"
        fi
        
        # 追加新的时间记录
        echo "scmiac,${ratio},${RUN_ELAPSED},${RUN_MINUTES}m ${RUN_SECONDS}s,$(date '+%Y-%m-%d %H:%M:%S')" >> "$time_file"
        echo ">>> Time saved to: $time_file"
        
        # 保存内存信息到 CSV
        memory_file="${time_dir}/memory.csv"
        
        # 如果文件不存在，创建表头
        if [ ! -f "$memory_file" ]; then
            echo "method,ratio,peak_memory_kb,peak_memory_mb,peak_memory_gb,timestamp" > "$memory_file"
        fi
        
        # 覆盖已有的 scmiac-ratio 记录（如果存在）
        if grep -q "^scmiac,${ratio}," "$memory_file" 2>/dev/null; then
            grep -v "^scmiac,${ratio}," "$memory_file" > "${memory_file}.tmp"
            mv "${memory_file}.tmp" "$memory_file"
        fi
        
        # 追加新的内存记录
        echo "scmiac,${ratio},${PEAK_MEMORY_KB},${PEAK_MEMORY_MB},${PEAK_MEMORY_GB},$(date '+%Y-%m-%d %H:%M:%S')" >> "$memory_file"
        echo ">>> Memory saved to: $memory_file"
    done
done

# 计算总耗时
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_ELAPSED / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "======================================"
echo "All experiments completed!"
echo "Total datasets: ${#DATASETS[@]}"
echo "Ratios per dataset: ${#PAIRED_RATIOS[@]}"
echo "Total runs: $((${#DATASETS[@]} * ${#PAIRED_RATIOS[@]}))"
echo "Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Finish time: $(date)"
echo "======================================"
