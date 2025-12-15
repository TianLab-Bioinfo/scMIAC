#!/bin/bash
# w/o Cell Type Filtering 消融实验
# 禁用细胞类型过滤，直接使用 MNN 锚点
#
# Usage:
#   bash wo_ctf.sh                              # 运行所有数据集
#   bash wo_ctf.sh 10x share                    # 运行指定数据集
#   bash wo_ctf.sh 10x --num-epochs 5           # 指定数据集 + 自定义参数
#   bash wo_ctf.sh --num-epochs 5 --print-step 1 # 所有数据集 + 自定义参数

set -e

# 设置时区为中国标准时间
export TZ='Asia/Shanghai'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMPL_SCRIPT="${PROJECT_ROOT}/scripts/run_methods/implementations/scmiac/main.py"

ALL_DATASETS=("brain" "kidney" "LungDroplet" "10x" "share" "wilk" "zhu")

# 分离数据集和额外参数
DATASETS=()
EXTRA_ARGS=()

for arg in "$@"; do
    # 如果参数以 - 开头，认为是额外参数
    if [[ "$arg" == -* ]]; then
        EXTRA_ARGS+=("$arg")
    else
        # 检查是否是已知的数据集名称
        if [[ " ${ALL_DATASETS[*]} " =~ " ${arg} " ]]; then
            DATASETS+=("$arg")
        else
            # 不是数据集名称，当作额外参数
            EXTRA_ARGS+=("$arg")
        fi
    fi
done

# 如果没有指定数据集，使用所有数据集
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("${ALL_DATASETS[@]}")
fi

# 激活 conda 环境
echo "Activating scMIAC conda environment..."
source /usr/local/opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/nfs_share/tlj/envs/scMIAC/

echo "======================================"
echo "Running w/o Cell Type Filtering ablation"
echo "Datasets: ${DATASETS[*]}"
echo "Using default configuration from scmiac package"
echo "Modification: --no-ct-filter"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra arguments: ${EXTRA_ARGS[*]}"
fi
echo "====================================="

for dataset in ${DATASETS[@]}; do
    output_dir="${PROJECT_ROOT}/data/${dataset}/output/ablation/wo_ctf"
    
    # 统计文件保存到 ablation/ 目录（每个数据集独立）
    stats_dir="${PROJECT_ROOT}/data/${dataset}/output/ablation"
    GPU_CSV="${stats_dir}/gpu.csv"
    
    echo ""
    echo ">>> Dataset: ${dataset}"
    echo ">>> Output: ${output_dir}"
    echo ">>> Start time: $(date)"
    
    # 记录开始时间
    DATASET_START=$(date +%s)
    
    # 使用 /usr/bin/time 监控内存使用
    TIME_LOG=$(mktemp)
    EXIT_CODE=0
    /usr/bin/time -v python "$IMPL_SCRIPT" \
        --dataset "${dataset}" \
        --data-root "${PROJECT_ROOT}/data" \
        --output-dir "${output_dir}" \
        --no-ct-filter \
        --identifier "wo_ctf" \
        --gpu-csv "${GPU_CSV}" \
        "${EXTRA_ARGS[@]}" \
        2> "$TIME_LOG" || EXIT_CODE=$?
    
    # 检查执行是否成功
    if [ $EXIT_CODE -ne 0 ]; then
        echo ">>> ERROR: Failed to run wo_ctf on dataset ${dataset} (exit code: $EXIT_CODE)" >&2
        echo ">>> See error details above" >&2
        cat "$TIME_LOG" >&2
        rm -f "$TIME_LOG"
        exit $EXIT_CODE
    fi
    
    # 计算耗时
    DATASET_END=$(date +%s)
    DATASET_ELAPSED=$((DATASET_END - DATASET_START))
    DATASET_MINUTES=$((DATASET_ELAPSED / 60))
    DATASET_SECONDS=$((DATASET_ELAPSED % 60))
    
    # 提取峰值内存
    PEAK_MEMORY_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
    if [ -z "$PEAK_MEMORY_KB" ]; then
        PEAK_MEMORY_KB=0
    fi
    PEAK_MEMORY_MB=$(echo "scale=2; $PEAK_MEMORY_KB / 1024" | bc)
    PEAK_MEMORY_GB=$(echo "scale=3; $PEAK_MEMORY_KB / 1024 / 1024" | bc)
    rm -f "$TIME_LOG"
    
    echo ">>> Completed: ${dataset} ($(date))"
    echo ">>> Elapsed time: ${DATASET_MINUTES}m ${DATASET_SECONDS}s"
    echo ">>> Peak memory: ${PEAK_MEMORY_MB} MB (${PEAK_MEMORY_GB} GB)"
    
    # 保存时间信息
    time_file="${stats_dir}/time.csv"
    mkdir -p "$stats_dir"
    if [ ! -f "$time_file" ]; then
        echo "method,elapsed_seconds,elapsed_time,timestamp" > "$time_file"
    fi
    if grep -q "^wo_ctf," "$time_file" 2>/dev/null; then
        grep -v "^wo_ctf," "$time_file" > "${time_file}.tmp"
        mv "${time_file}.tmp" "$time_file"
    fi
    echo "wo_ctf,${DATASET_ELAPSED},${DATASET_MINUTES}m ${DATASET_SECONDS}s,$(date '+%Y-%m-%d %H:%M:%S')" >> "$time_file"
    echo ">>> Time saved to: $time_file"
    
    # 保存内存信息
    memory_file="${stats_dir}/memory.csv"
    if [ ! -f "$memory_file" ]; then
        echo "method,peak_memory_kb,peak_memory_mb,peak_memory_gb,timestamp" > "$memory_file"
    fi
    if grep -q "^wo_ctf," "$memory_file" 2>/dev/null; then
        grep -v "^wo_ctf," "$memory_file" > "${memory_file}.tmp"
        mv "${memory_file}.tmp" "$memory_file"
    fi
    echo "wo_ctf,${PEAK_MEMORY_KB},${PEAK_MEMORY_MB},${PEAK_MEMORY_GB},$(date '+%Y-%m-%d %H:%M:%S')" >> "$memory_file"
    echo ">>> Memory saved to: $memory_file"
done

echo ""
echo "======================================"
echo "w/o Cell Type Filtering ablation completed!"
echo "Total datasets: ${#DATASETS[@]}"
echo "Finish time: $(date)"
echo "======================================"
