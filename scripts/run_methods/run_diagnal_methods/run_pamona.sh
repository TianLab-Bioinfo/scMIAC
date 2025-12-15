#!/bin/bash
# PAMONA 方法批量运行脚本 - 对角线整合测评
# 对角线整合测评 - 循环运行所有数据集进行基准测试
#
# Usage:
#   bash run_pamona.sh                    # 运行所有数据集（使用细胞类型先验）
#   bash run_pamona.sh 10x share          # 运行指定数据集
#   bash run_pamona.sh --no-prior 10x     # 禁用细胞类型先验
#   bash run_pamona.sh --help             # 显示帮助

set -e

# 设置时区为中国标准时间
export TZ='Asia/Shanghai'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMPL_SCRIPT="${SCRIPT_DIR}/../implementations/pamona/main.py"

# 所有数据集
ALL_DATASETS=("kidney" "LungDroplet" "10x" "share" "wilk" "zhu")

# 显示帮助信息
show_help() {
    cat << EOF
PAMONA 方法批量运行脚本 - 对角线整合测评

Usage:
  bash run_pamona.sh [datasets...] [extra_args...]
  
  datasets     指定要运行的数据集（可选，默认运行所有）
  extra_args   传递给 pamona 脚本的额外参数

Available datasets: ${ALL_DATASETS[*]}

Examples:
  bash run_pamona.sh                                      # 运行所有数据集（默认使用先验）
  bash run_pamona.sh 10x share                            # 只运行 10x 和 share
  bash run_pamona.sh --no-prior                           # 禁用先验运行所有数据集
  bash run_pamona.sh --no-prior 10x                       # 禁用先验运行 10x
  bash run_pamona.sh --rna-celltype-key custom_rna 10x    # 自定义RNA列名
  bash run_pamona.sh --atac-celltype-key custom_atac 10x  # 自定义ATAC列名
  bash run_pamona.sh 10x --seed 42                        # 自定义随机种子

Common Options:
  --no-prior                     禁用细胞类型先验信息（默认使用先验）
  --rna-celltype-key <key>       RNA细胞类型列名（默认: cell_type）
  --atac-celltype-key <key>      ATAC细胞类型列名（默认: pred）
  --seed <seed>                  随机种子（默认: 24）

Output:
  With prior (default): data/<dataset>/output/diagnal_methods/pamona/
  Without prior (--no-prior): data/<dataset>/output/diagnal_methods/pamona_noct/

EOF
}

# 处理参数：分离数据集和额外参数
DATASETS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        show_help
        exit 0
    elif [[ "$1" == -* ]]; then
        # 参数以 - 开头，认为是额外参数
        EXTRA_ARGS+=("$1")
        shift
    else
        # 检查是否是已知的数据集名称
        if [[ " ${ALL_DATASETS[*]} " =~ " ${1} " ]]; then
            DATASETS+=("$1")
            shift
        else
            # 不是数据集名称，可能是某个参数的值，当作额外参数
            EXTRA_ARGS+=("$1")
            shift
        fi
    fi
done

# 如果没有指定数据集，使用所有数据集
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("${ALL_DATASETS[@]}")
fi

# 激活 conda 环境
echo "Activating numpyv1 conda environment..."
source /usr/local/opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/nfs_share/tlj/envs/numpyv1/

echo "======================================"
echo "Running PAMONA on datasets: ${DATASETS[*]}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra arguments: ${EXTRA_ARGS[*]}"
fi
echo "======================================"

# 记录总开始时间
TOTAL_START=$(date +%s)

# 运行每个数据集
for dataset in "${DATASETS[@]}"; do
    data_root="${PROJECT_ROOT}/data"
    
    # 根据是否使用 --no-prior 选择输出目录
    if [[ " ${EXTRA_ARGS[*]} " =~ " --no-prior " ]]; then
        output_dir="${data_root}/${dataset}/output/diagnal_methods/pamona_noct"
    else
        output_dir="${data_root}/${dataset}/output/diagnal_methods/pamona"
    fi
    
    echo ""
    echo ">>> Dataset: $dataset"
    echo ">>> Data root: $data_root"
    echo ">>> Output: $output_dir"
    echo ">>> Start time: $(date)"
    
    # 记录数据集开始时间
    DATASET_START=$(date +%s)
    
    # 使用 /usr/bin/time 监控内存使用
    TIME_LOG=$(mktemp)
    EXIT_CODE=0
    /usr/bin/time -v python "$IMPL_SCRIPT" \
        --dataset "$dataset" \
        --data-root "$data_root" \
        --output-dir "$output_dir" \
        "${EXTRA_ARGS[@]}" \
        2> "$TIME_LOG" || EXIT_CODE=$?
    
    # 检查执行是否成功
    if [ $EXIT_CODE -ne 0 ]; then
        echo ">>> ERROR: Failed to run pamona on dataset $dataset (exit code: $EXIT_CODE)" >&2
        echo ">>> See error details above" >&2
        cat "$TIME_LOG" >&2
        rm -f "$TIME_LOG"
        exit $EXIT_CODE
    fi
    
    # 计算数据集耗时
    DATASET_END=$(date +%s)
    DATASET_ELAPSED=$((DATASET_END - DATASET_START))
    DATASET_MINUTES=$((DATASET_ELAPSED / 60))
    DATASET_SECONDS=$((DATASET_ELAPSED % 60))
    
    # 提取峰值内存 (Maximum resident set size in KB)
    PEAK_MEMORY_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
    if [ -z "$PEAK_MEMORY_KB" ]; then
        PEAK_MEMORY_KB=0
    fi
    PEAK_MEMORY_MB=$(echo "scale=2; $PEAK_MEMORY_KB / 1024" | bc)
    PEAK_MEMORY_GB=$(echo "scale=3; $PEAK_MEMORY_KB / 1024 / 1024" | bc)
    rm -f "$TIME_LOG"
    
    echo ">>> Completed: $dataset ($(date))"
    echo ">>> Elapsed time: ${DATASET_MINUTES}m ${DATASET_SECONDS}s"
    echo ">>> Peak memory: ${PEAK_MEMORY_MB} MB (${PEAK_MEMORY_GB} GB)"
    
    # 保存时间信息到 CSV
    time_dir="${data_root}/${dataset}/output/diagnal_methods"
    time_file="${time_dir}/time.csv"
    mkdir -p "$time_dir"
    
    # 如果文件不存在，创建表头
    if [ ! -f "$time_file" ]; then
        echo "method,elapsed_seconds,elapsed_time,timestamp" > "$time_file"
    fi
    
    # 确定方法名（根据是否使用先验）
    if [[ " ${EXTRA_ARGS[*]} " =~ " --no-prior " ]]; then
        method_name="pamona_noct"
    else
        method_name="pamona"
    fi
    
    # 覆盖已有的该方法记录（如果存在）
    if grep -q "^${method_name}," "$time_file" 2>/dev/null; then
        # 使用临时文件过滤掉旧记录
        grep -v "^${method_name}," "$time_file" > "${time_file}.tmp"
        mv "${time_file}.tmp" "$time_file"
    fi
    
    # 追加新的时间记录
    echo "${method_name},${DATASET_ELAPSED},${DATASET_MINUTES}m ${DATASET_SECONDS}s,$(date '+%Y-%m-%d %H:%M:%S')" >> "$time_file"
    echo ">>> Time saved to: $time_file"
    
    # 保存内存信息到 CSV
    memory_file="${time_dir}/memory.csv"
    
    # 如果文件不存在，创建表头
    if [ ! -f "$memory_file" ]; then
        echo "method,peak_memory_kb,peak_memory_mb,peak_memory_gb,timestamp" > "$memory_file"
    fi
    
    # 覆盖已有的该方法记录（如果存在）
    if grep -q "^${method_name}," "$memory_file" 2>/dev/null; then
        # 使用临时文件过滤掉旧记录
        grep -v "^${method_name}," "$memory_file" > "${memory_file}.tmp"
        mv "${memory_file}.tmp" "$memory_file"
    fi
    
    # 追加新的内存记录
    echo "${method_name},${PEAK_MEMORY_KB},${PEAK_MEMORY_MB},${PEAK_MEMORY_GB},$(date '+%Y-%m-%d %H:%M:%S')" >> "$memory_file"
    echo ">>> Memory saved to: $memory_file"
done

# 计算总耗时
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_ELAPSED / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "======================================"
echo "All datasets completed!"
echo "Total datasets: ${#DATASETS[@]}"
echo "Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Finish time: $(date)"
echo "====================================="
