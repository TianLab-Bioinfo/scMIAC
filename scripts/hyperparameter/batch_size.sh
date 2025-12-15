#!/bin/bash
# Batch Size 超参数实验（仅在 10x 数据集上运行）
# 测试不同批量大小对模型性能的影响
#
# Usage:
#   bash batch_size.sh                              # 运行实验
#   bash batch_size.sh --num-epochs 5 --print-step 1 # 自定义参数

set -e

# 设置时区为中国标准时间
export TZ='Asia/Shanghai'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMPL_SCRIPT="${PROJECT_ROOT}/scripts/run_methods/implementations/scmiac/main.py"

DATASET="10x"  # 超参数实验固定使用 10x 数据集
BATCH_SIZES=(64 256 1024 4096 8192)  # 默认值: 1024

# 激活 conda 环境
echo "Activating scMIAC conda environment..."
source /usr/local/opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/nfs_share/tlj/envs/scMIAC/

echo "======================================"
echo "Running Batch Size hyperparameter sweep"
echo "Dataset: ${DATASET}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
if [ $# -gt 0 ]; then
    echo "Extra arguments: $@"
fi
echo "====================================="

# 记录总开始时间
TOTAL_START=$(date +%s)

for batch_size in "${BATCH_SIZES[@]}"; do
    output_dir="${PROJECT_ROOT}/data/${DATASET}/output/hyperparameter/batch_size/${batch_size}"
    
    # 统计文件保存到 hyperparameter/batch_size/ 目录（所有batch_size值共享）
    stats_dir="${PROJECT_ROOT}/data/${DATASET}/output/hyperparameter/batch_size"
    GPU_CSV_PATH="${stats_dir}/gpu.csv"
    
    echo ""
    echo ">>> Batch size: ${batch_size}"
    echo ">>> Output: ${output_dir}"
    echo ">>> Start time: $(date)"
    
    # 记录开始时间
    BATCH_START=$(date +%s)
    
    # 使用 /usr/bin/time 监控内存使用
    TIME_LOG=$(mktemp)
    EXIT_CODE=0
    /usr/bin/time -v python -u "$IMPL_SCRIPT" \
        --dataset "${DATASET}" \
        --data-root "${PROJECT_ROOT}/data" \
        --output-dir "${output_dir}" \
        --batch-size ${batch_size} \
        --identifier "${batch_size}" \
        --gpu-csv "${GPU_CSV_PATH}" \
        "$@" \
        2> "$TIME_LOG" || EXIT_CODE=$?
    
    # 检查执行是否成功
    if [ $EXIT_CODE -ne 0 ]; then
        echo ">>> ERROR: Failed to run batch_size=${batch_size} (exit code: $EXIT_CODE)" >&2
        echo ">>> See error details above" >&2
        cat "$TIME_LOG" >&2
        rm -f "$TIME_LOG"
        exit $EXIT_CODE
    fi
    
    # 计算耗时
    BATCH_END=$(date +%s)
    BATCH_ELAPSED=$((BATCH_END - BATCH_START))
    BATCH_MINUTES=$((BATCH_ELAPSED / 60))
    BATCH_SECONDS=$((BATCH_ELAPSED % 60))
    
    # 提取峰值内存 (Maximum resident set size in KB)
    PEAK_MEMORY_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
    if [ -z "$PEAK_MEMORY_KB" ]; then
        PEAK_MEMORY_KB=0
    fi
    PEAK_MEMORY_MB=$(echo "scale=2; $PEAK_MEMORY_KB / 1024" | bc)
    PEAK_MEMORY_GB=$(echo "scale=3; $PEAK_MEMORY_KB / 1024 / 1024" | bc)
    rm -f "$TIME_LOG"
    
    echo ">>> Completed: batch_size=${batch_size} ($(date))"
    echo ">>> Elapsed time: ${BATCH_MINUTES}m ${BATCH_SECONDS}s"
    echo ">>> Peak memory: ${PEAK_MEMORY_MB} MB (${PEAK_MEMORY_GB} GB)"
    
    # 保存时间信息到 CSV
    stats_dir="${PROJECT_ROOT}/data/${DATASET}/output/hyperparameter/batch_size"
    time_file="${stats_dir}/time.csv"
    mkdir -p "$stats_dir"
    
    # 如果文件不存在，创建表头
    if [ ! -f "$time_file" ]; then
        echo "batch_size,elapsed_seconds,elapsed_time,timestamp" > "$time_file"
    fi
    
    # 覆盖已有的 batch_size 记录（如果存在）
    if grep -q "^${batch_size}," "$time_file" 2>/dev/null; then
        # 使用临时文件过滤掉旧记录
        grep -v "^${batch_size}," "$time_file" > "${time_file}.tmp"
        mv "${time_file}.tmp" "$time_file"
    fi
    
    # 追加新的时间记录
    echo "${batch_size},${BATCH_ELAPSED},${BATCH_MINUTES}m ${BATCH_SECONDS}s,$(date '+%Y-%m-%d %H:%M:%S')" >> "$time_file"
    echo ">>> Time saved to: $time_file"
    
    # 保存内存信息到 CSV
    memory_file="${stats_dir}/memory.csv"
    
    # 如果文件不存在，创建表头
    if [ ! -f "$memory_file" ]; then
        echo "batch_size,peak_memory_kb,peak_memory_mb,peak_memory_gb,timestamp" > "$memory_file"
    fi
    
    # 覆盖已有的 batch_size 记录（如果存在）
    if grep -q "^${batch_size}," "$memory_file" 2>/dev/null; then
        # 使用临时文件过滤掉旧记录
        grep -v "^${batch_size}," "$memory_file" > "${memory_file}.tmp"
        mv "${memory_file}.tmp" "$memory_file"
    fi
    
    # 追加新的内存记录
    echo "${batch_size},${PEAK_MEMORY_KB},${PEAK_MEMORY_MB},${PEAK_MEMORY_GB},$(date '+%Y-%m-%d %H:%M:%S')" >> "$memory_file"
    echo ">>> Memory saved to: $memory_file"
done

# 计算总耗时
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_ELAPSED / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "======================================"
echo "Batch Size hyperparameter sweep completed!"
echo "Total experiments: ${#BATCH_SIZES[@]}"
echo "Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Finish time: $(date)"
echo "======================================"
