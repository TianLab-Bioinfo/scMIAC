#!/bin/bash
# Annotation Accuracy 超参数实验
# 测试细胞类型注释准确率对整合质量的影响
#
# 通过 --anno-accuracy 参数动态注入注释噪声，无需预先生成数据集
#
# Usage:
#   bash anno_acc.sh                                    # 默认运行 10x，所有准确率
#   bash anno_acc.sh share                              # 运行 share 数据集，所有准确率
#   bash anno_acc.sh 10x share wilk                     # 运行多个数据集，所有准确率
#   bash anno_acc.sh --anno-accs 5 10                   # 默认10x，只运行 5% 和 10%
#   bash anno_acc.sh share --anno-accs 5                # share 数据集，只运行 5%
#   bash anno_acc.sh --num-epochs 5 --print-step 1     # 自定义参数（使用默认10x和所有准确率）
#   bash anno_acc.sh share --anno-accs 5 10 --num-epochs 5  # 组合使用

set -e

# 设置时区为中国标准时间
export TZ='Asia/Shanghai'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMPL_SCRIPT="${PROJECT_ROOT}/scripts/run_methods/implementations/scmiac/main.py"

ALL_DATASETS=("10x" "share" "brain" "kidney" "LungDroplet" "wilk" "zhu")
DEFAULT_ANNO_ACCS=(100 90 80 70 60 50 20 10 5)

# 分离数据集、anno_accs 和额外参数
DATASETS=()
ANNO_ACCS=()
EXTRA_ARGS=()
PARSING_ANNO_ACCS=false

for arg in "$@"; do
    # 如果遇到 --anno-accs，开始解析 anno_accs
    if [[ "$arg" == "--anno-accs" ]]; then
        PARSING_ANNO_ACCS=true
        continue
    fi
    
    # 如果正在解析 anno_accs
    if [[ "$PARSING_ANNO_ACCS" == true ]]; then
        # 如果遇到其他 -- 参数，停止解析 anno_accs
        if [[ "$arg" == --* ]]; then
            PARSING_ANNO_ACCS=false
            EXTRA_ARGS+=("$arg")
        # 如果是数字，添加到 ANNO_ACCS
        elif [[ "$arg" =~ ^[0-9]+$ ]]; then
            ANNO_ACCS+=("$arg")
        else
            # 不是数字，可能是数据集名称或其他参数
            PARSING_ANNO_ACCS=false
            if [[ " ${ALL_DATASETS[*]} " =~ " ${arg} " ]]; then
                DATASETS+=("$arg")
            else
                EXTRA_ARGS+=("$arg")
            fi
        fi
    else
        # 检查参数是否以 - 或 -- 开头
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
    fi
done

# 如果没有指定数据集，默认使用 10x（保持向后兼容）
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("10x")
fi

# 如果没有指定 anno_accs，使用默认值
if [ ${#ANNO_ACCS[@]} -eq 0 ]; then
    ANNO_ACCS=("${DEFAULT_ANNO_ACCS[@]}")
fi

# 激活 conda 环境
echo "Activating scMIAC conda environment..."
source /usr/local/opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/nfs_share/tlj/envs/scMIAC/

echo "======================================"
echo "Running Annotation Accuracy hyperparameter sweep"
echo "Datasets: ${DATASETS[*]}"
echo "Annotation accuracies: ${ANNO_ACCS[*]}%"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra arguments: ${EXTRA_ARGS[*]}"
fi
echo "====================================="
echo ""
echo "Note: Annotation noise will be injected dynamically during training"
echo "      100% accuracy uses original labels; others randomly replace labels with other cell types"
echo ""

# 记录总开始时间
TOTAL_START=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "######################################"
    echo "### Processing dataset: ${dataset}"
    echo "######################################"
    echo ""
    
    # 统计文件保存目录
    stats_dir="${PROJECT_ROOT}/data/${dataset}/output/hyperparameter/anno_acc"
    GPU_CSV_PATH="${stats_dir}/gpu.csv"
    
    # 记录数据集开始时间
    DATASET_START=$(date +%s)

    for anno_acc in "${ANNO_ACCS[@]}"; do
        output_dir="${PROJECT_ROOT}/data/${dataset}/output/hyperparameter/anno_acc/${anno_acc}"
    
        echo ""
        echo ">>> Dataset: ${dataset} | Annotation accuracy: ${anno_acc}%"
        echo ">>> Output: ${output_dir}"
        echo ">>> Start time: $(date)"
        
        # 记录开始时间
        ANNO_START=$(date +%s)
        
        # 使用 /usr/bin/time 监控内存使用
        TIME_LOG=$(mktemp)
        EXIT_CODE=0
        
        # 构建命令：100%不需要--anno-accuracy参数，其他需要
        if [ "${anno_acc}" -eq 100 ]; then
            /usr/bin/time -v python -u "$IMPL_SCRIPT" \
                --dataset "${dataset}" \
                --data-root "${PROJECT_ROOT}/data" \
                --output-dir "${output_dir}" \
                --identifier "${anno_acc}" \
                --rna-celltype-key cell_type_merge \
                --atac-celltype-key cell_type_merge \
                --gpu-csv "${GPU_CSV_PATH}" \
                "${EXTRA_ARGS[@]}" \
                2> "$TIME_LOG" || EXIT_CODE=$?
        else
            /usr/bin/time -v python -u "$IMPL_SCRIPT" \
                --dataset "${dataset}" \
                --data-root "${PROJECT_ROOT}/data" \
                --output-dir "${output_dir}" \
                --anno-accuracy "${anno_acc}" \
                --identifier "${anno_acc}" \
                --rna-celltype-key cell_type_merge \
                --atac-celltype-key cell_type_merge \
                --gpu-csv "${GPU_CSV_PATH}" \
                "${EXTRA_ARGS[@]}" \
                2> "$TIME_LOG" || EXIT_CODE=$?
        fi
    
        # 检查执行是否成功
        if [ $EXIT_CODE -ne 0 ]; then
            echo ">>> ERROR: Failed to run ${dataset} anno_acc=${anno_acc}% (exit code: $EXIT_CODE)" >&2
            echo ">>> See error details above" >&2
            cat "$TIME_LOG" >&2
            rm -f "$TIME_LOG"
            exit $EXIT_CODE
        fi
        
        # 计算耗时
        ANNO_END=$(date +%s)
        ANNO_ELAPSED=$((ANNO_END - ANNO_START))
        ANNO_MINUTES=$((ANNO_ELAPSED / 60))
        ANNO_SECONDS=$((ANNO_ELAPSED % 60))
        
        # 提取峰值内存 (Maximum resident set size in KB)
        PEAK_MEMORY_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
        if [ -z "$PEAK_MEMORY_KB" ]; then
            PEAK_MEMORY_KB=0
        fi
        PEAK_MEMORY_MB=$(echo "scale=2; $PEAK_MEMORY_KB / 1024" | bc)
        PEAK_MEMORY_GB=$(echo "scale=3; $PEAK_MEMORY_KB / 1024 / 1024" | bc)
        rm -f "$TIME_LOG"
        
        echo ">>> Completed: ${dataset} anno_acc=${anno_acc}% ($(date))"
        echo ">>> Elapsed time: ${ANNO_MINUTES}m ${ANNO_SECONDS}s"
        echo ">>> Peak memory: ${PEAK_MEMORY_MB} MB (${PEAK_MEMORY_GB} GB)"
        
        # 保存时间信息到 CSV
        time_file="${stats_dir}/time.csv"
        mkdir -p "$stats_dir"
        
        # 如果文件不存在，创建表头
        if [ ! -f "$time_file" ]; then
            echo "anno_acc,elapsed_seconds,elapsed_time,timestamp" > "$time_file"
        fi
        
        # 覆盖已有的 anno_acc 记录（如果存在）
        if grep -q "^${anno_acc}," "$time_file" 2>/dev/null; then
            grep -v "^${anno_acc}," "$time_file" > "${time_file}.tmp"
            mv "${time_file}.tmp" "$time_file"
        fi
        
        # 追加新的时间记录
        echo "${anno_acc},${ANNO_ELAPSED},${ANNO_MINUTES}m ${ANNO_SECONDS}s,$(date '+%Y-%m-%d %H:%M:%S')" >> "$time_file"
        echo ">>> Time saved to: $time_file"
        
        # 保存内存信息到 CSV
        memory_file="${stats_dir}/memory.csv"
        
        # 如果文件不存在，创建表头
        if [ ! -f "$memory_file" ]; then
            echo "anno_acc,peak_memory_kb,peak_memory_mb,peak_memory_gb,timestamp" > "$memory_file"
        fi
        
        # 覆盖已有的 anno_acc 记录（如果存在）
        if grep -q "^${anno_acc}," "$memory_file" 2>/dev/null; then
            grep -v "^${anno_acc}," "$memory_file" > "${memory_file}.tmp"
            mv "${memory_file}.tmp" "$memory_file"
        fi
        
        # 追加新的内存记录
        echo "${anno_acc},${PEAK_MEMORY_KB},${PEAK_MEMORY_MB},${PEAK_MEMORY_GB},$(date '+%Y-%m-%d %H:%M:%S')" >> "$memory_file"
        echo ">>> Memory saved to: $memory_file"
    done
    
    # 计算数据集总耗时
    DATASET_END=$(date +%s)
    DATASET_ELAPSED=$((DATASET_END - DATASET_START))
    DATASET_MINUTES=$((DATASET_ELAPSED / 60))
    DATASET_SECONDS=$((DATASET_ELAPSED % 60))
    
    echo ""
    echo ">>> Dataset ${dataset} completed in ${DATASET_MINUTES}m ${DATASET_SECONDS}s"
    echo ""
done

# 计算总耗时
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_ELAPSED / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "======================================"
echo "Annotation Accuracy hyperparameter sweep completed!"
echo "Total datasets: ${#DATASETS[@]}"
echo "Datasets: ${DATASETS[*]}"
echo "Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Finish time: $(date)"
echo "======================================"
