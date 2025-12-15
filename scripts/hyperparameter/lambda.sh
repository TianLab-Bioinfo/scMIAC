#!/bin/bash
# Lambda Contra 超参数实验（仅在 10x 数据集上运行）
# 测试对比学习损失权重对模态混合的影响
#
# Usage:
#   bash lambda.sh                              # 运行实验
#   bash lambda.sh --num-epochs 5 --print-step 1 # 自定义参数

set -e

# 设置时区为中国标准时间
export TZ='Asia/Shanghai'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMPL_SCRIPT="${PROJECT_ROOT}/scripts/run_methods/implementations/scmiac/main.py"

DATASET="10x"  # 超参数实验固定使用 10x 数据集
LAMBDA_VALUES=(1 10 100 1000 10000)  

# 激活 conda 环境
echo "Activating scMIAC conda environment..."
source /usr/local/opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/nfs_share/tlj/envs/scMIAC/

echo "======================================"
echo "Running Lambda Contra hyperparameter sweep"
echo "Dataset: ${DATASET}"
echo "Lambda values: ${LAMBDA_VALUES[*]}"
if [ $# -gt 0 ]; then
    echo "Extra arguments: $@"
fi
echo "====================================="

for lambda_contra in "${LAMBDA_VALUES[@]}"; do
    output_dir="${PROJECT_ROOT}/data/${DATASET}/output/hyperparameter/lambda/${lambda_contra}"
    
    # 统计文件保存到 hyperparameter/lambda/ 目录（所有lambda值共享）
    stats_dir="${PROJECT_ROOT}/data/${DATASET}/output/hyperparameter/lambda"
    GPU_CSV_PATH="${stats_dir}/gpu.csv"
    
    echo ""
    echo ">>> Lambda contra: ${lambda_contra}"
    echo ">>> Output: ${output_dir}"
    echo ">>> Start time: $(date)"
    
    # 记录开始时间
    LAMBDA_START=$(date +%s)
    
    # 使用 /usr/bin/time 监控内存使用
    TIME_LOG=$(mktemp)
    EXIT_CODE=0
    /usr/bin/time -v python -u "$IMPL_SCRIPT" \
        --dataset "${DATASET}" \
        --data-root "${PROJECT_ROOT}/data" \
        --output-dir "${output_dir}" \
        --lambda-contra ${lambda_contra} \
        --identifier "${lambda_contra}" \
        --gpu-csv "${GPU_CSV_PATH}" \
        "$@" \
        2> "$TIME_LOG" || EXIT_CODE=$?
    
    # 检查执行是否成功
    if [ $EXIT_CODE -ne 0 ]; then
        echo ">>> ERROR: Failed to run lambda_contra=${lambda_contra} (exit code: $EXIT_CODE)" >&2
        echo ">>> See error details above" >&2
        cat "$TIME_LOG" >&2
        rm -f "$TIME_LOG"
        exit $EXIT_CODE
    fi
    
    # 计算耗时
    LAMBDA_END=$(date +%s)
    LAMBDA_ELAPSED=$((LAMBDA_END - LAMBDA_START))
    LAMBDA_MINUTES=$((LAMBDA_ELAPSED / 60))
    LAMBDA_SECONDS=$((LAMBDA_ELAPSED % 60))
    
    # 提取峰值内存 (Maximum resident set size in KB)
    PEAK_MEMORY_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
    if [ -z "$PEAK_MEMORY_KB" ]; then
        PEAK_MEMORY_KB=0
    fi
    PEAK_MEMORY_MB=$(echo "scale=2; $PEAK_MEMORY_KB / 1024" | bc)
    PEAK_MEMORY_GB=$(echo "scale=3; $PEAK_MEMORY_KB / 1024 / 1024" | bc)
    rm -f "$TIME_LOG"
    
    echo ">>> Completed: lambda_contra=${lambda_contra} ($(date))"
    echo ">>> Elapsed time: ${LAMBDA_MINUTES}m ${LAMBDA_SECONDS}s"
    echo ">>> Peak memory: ${PEAK_MEMORY_MB} MB (${PEAK_MEMORY_GB} GB)"
    
    # 保存时间信息到 CSV
    time_file="${stats_dir}/time.csv"
    mkdir -p "$stats_dir"
    
    # 如果文件不存在，创建表头
    if [ ! -f "$time_file" ]; then
        echo "lambda_contra,elapsed_seconds,elapsed_time,timestamp" > "$time_file"
    fi
    
    # 覆盖已有的 lambda_contra 记录（如果存在）
    if grep -q "^${lambda_contra}," "$time_file" 2>/dev/null; then
        grep -v "^${lambda_contra}," "$time_file" > "${time_file}.tmp"
        mv "${time_file}.tmp" "$time_file"
    fi
    
    # 追加新的时间记录
    echo "${lambda_contra},${LAMBDA_ELAPSED},${LAMBDA_MINUTES}m ${LAMBDA_SECONDS}s,$(date '+%Y-%m-%d %H:%M:%S')" >> "$time_file"
    echo ">>> Time saved to: $time_file"
    
    # 保存内存信息到 CSV
    memory_file="${stats_dir}/memory.csv"
    
    # 如果文件不存在，创建表头
    if [ ! -f "$memory_file" ]; then
        echo "lambda_contra,peak_memory_kb,peak_memory_mb,peak_memory_gb,timestamp" > "$memory_file"
    fi
    
    # 覆盖已有的 lambda_contra 记录（如果存在）
    if grep -q "^${lambda_contra}," "$memory_file" 2>/dev/null; then
        grep -v "^${lambda_contra}," "$memory_file" > "${memory_file}.tmp"
        mv "${memory_file}.tmp" "$memory_file"
    fi
    
    # 追加新的内存记录
    echo "${lambda_contra},${PEAK_MEMORY_KB},${PEAK_MEMORY_MB},${PEAK_MEMORY_GB},$(date '+%Y-%m-%d %H:%M:%S')" >> "$memory_file"
    echo ">>> Memory saved to: $memory_file"
done

echo ""
echo "======================================"
echo "Lambda Contra hyperparameter sweep completed!"
echo "Total experiments: ${#LAMBDA_VALUES[@]}"
echo "Finish time: $(date)"
echo "======================================"
