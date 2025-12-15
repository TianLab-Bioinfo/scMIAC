#!/bin/bash
# scANVI 方法批量运行脚本 - 对角线整合测评
# 对角线整合测评 - 循环运行所有数据集进行基准测试
#
# Usage:
#   bash run_scanvi.sh                     # 运行所有数据集
#   bash run_scanvi.sh 10x share           # 运行指定数据集
#   bash run_scanvi.sh --device cpu 10x    # 使用 CPU
#   bash run_scanvi.sh --help              # 显示帮助

set -e

# 设置时区为中国标准时间
export TZ='Asia/Shanghai'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMPL_SCRIPT="${SCRIPT_DIR}/../implementations/scanvi/main.py"

# 所有数据集
ALL_DATASETS=("brain" "kidney" "LungDroplet" "10x" "share" "wilk" "zhu")

# 默认参数
DEVICE="cuda:0"
SCVI_EPOCHS=200
SCANVI_EPOCHS=20

# 显示帮助信息
show_help() {
    cat << EOF
scANVI 方法批量运行脚本 - 对角线整合测评

Usage:
  bash run_scanvi.sh [OPTIONS] [<dataset1> ...]  # 运行指定数据集
  bash run_scanvi.sh --help                       # 显示帮助

Options:
  --device <device>         设备选择 (default: cuda:0, 可选: cuda:1, cpu)
  --scvi-epochs <num>       scVI 训练轮数 (default: 200)
  --scanvi-epochs <num>     scANVI 训练轮数 (default: 20)

Available datasets: ${ALL_DATASETS[*]}

Examples:
  bash run_scanvi.sh                              # 运行所有数据集 (GPU)
  bash run_scanvi.sh 10x share                    # 只运行 10x 和 share
  bash run_scanvi.sh --device cpu 10x             # 使用 CPU 运行 10x
  bash run_scanvi.sh --scvi-epochs 100 10x        # 自定义训练轮数

Output:
  - data/<dataset>/output/diagnal_methods/scanvi/

Files generated:
  - rna_embeddings.csv         RNA 模态嵌入向量
  - atac_embeddings.csv        ATAC 模态嵌入向量
  - scanvi_latent_umap.png     UMAP 可视化
  - combined.h5ad              合并的 AnnData 对象
  - time.csv                   运行时间统计
  - memory.csv                 内存使用统计

EOF
}

# 处理参数
DATASETS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --scvi-epochs)
            SCVI_EPOCHS="$2"
            shift 2
            ;;
        --scanvi-epochs)
            SCANVI_EPOCHS="$2"
            shift 2
            ;;
        *)
            DATASETS+=("$1")
            shift
            ;;
    esac
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
echo "Running scANVI on datasets: ${DATASETS[*]}"
echo "Device: $DEVICE"
echo "scVI epochs: $SCVI_EPOCHS"
echo "scANVI epochs: $SCANVI_EPOCHS"
echo "======================================"

# 记录总开始时间
TOTAL_START=$(date +%s)

# 运行每个数据集
for dataset in "${DATASETS[@]}"; do
    data_root="${PROJECT_ROOT}/data"
    output_dir="${data_root}/${dataset}/output/diagnal_methods/scanvi"
    
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
        --device "$DEVICE" \
        --scvi-epochs "$SCVI_EPOCHS" \
        --scanvi-epochs "$SCANVI_EPOCHS" \
        2> "$TIME_LOG" || EXIT_CODE=$?
    
    # 检查执行是否成功
    if [ $EXIT_CODE -ne 0 ]; then
        echo ">>> ERROR: Failed to run scANVI on dataset $dataset (exit code: $EXIT_CODE)" >&2
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
    
    # 覆盖已有的该方法记录（如果存在）
    if grep -q "^scanvi," "$time_file" 2>/dev/null; then
        # 使用临时文件过滤掉旧记录
        grep -v "^scanvi," "$time_file" > "${time_file}.tmp"
        mv "${time_file}.tmp" "$time_file"
    fi
    
    # 追加新的时间记录
    echo "scanvi,${DATASET_ELAPSED},${DATASET_MINUTES}m ${DATASET_SECONDS}s,$(date '+%Y-%m-%d %H:%M:%S')" >> "$time_file"
    echo ">>> Time saved to: $time_file"
    
    # 保存内存信息到 CSV
    memory_file="${time_dir}/memory.csv"
    
    # 如果文件不存在，创建表头
    if [ ! -f "$memory_file" ]; then
        echo "method,peak_memory_kb,peak_memory_mb,peak_memory_gb,timestamp" > "$memory_file"
    fi
    
    # 覆盖已有的该方法记录（如果存在）
    if grep -q "^scanvi," "$memory_file" 2>/dev/null; then
        # 使用临时文件过滤掉旧记录
        grep -v "^scanvi," "$memory_file" > "${memory_file}.tmp"
        mv "${memory_file}.tmp" "$memory_file"
    fi
    
    # 追加新的内存记录
    echo "scanvi,${PEAK_MEMORY_KB},${PEAK_MEMORY_MB},${PEAK_MEMORY_GB},$(date '+%Y-%m-%d %H:%M:%S')" >> "$memory_file"
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
echo "======================================"
