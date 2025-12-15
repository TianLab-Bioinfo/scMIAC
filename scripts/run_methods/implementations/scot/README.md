# SCOT (Single-Cell Optimal Transport) Module

这个模块包含 SCOT 系列算法的实现，用于单细胞多模态数据整合。

## 文件说明

- `main.py` - SCOTv2 方法的主运行脚本
- `scotv2.py` - SCOTv2 算法核心实现（支持 CPU/GPU）
- `scotv1.py` - SCOTv1 算法实现（原始版本）
- `evals.py` - 评估函数和工具
- `__init__.py` - Python 包初始化文件

## 使用方法

### 基本用法

#### CPU 运行（默认）
```bash
# 直接运行 Python 脚本
python main.py --dataset 10x --output-dir output/scotv2

# 通过批量运行脚本
bash ../../run/run_scotv2.sh
bash ../../run/run_scotv2.sh 10x share
```

#### GPU 加速运行
```bash
# 直接运行 Python 脚本（需要 CUDA 可用）
python main.py --dataset share --output-dir output/scotv2 --device cuda

# 通过批量运行脚本
bash ../../run/run_scotv2.sh --device cuda
bash ../../run/run_scotv2.sh --device cuda share
```

### 完整命令行参数

```bash
python main.py \
    --dataset DATASET        # 数据集名称（必需）
    --output-dir DIR         # 输出目录（必需）
    --data-root DIR          # 数据根目录（默认: data）
    --seed INT               # 随机种子（默认: 24）
    --device {cpu,cuda}      # 计算设备（默认: cpu）
```

### SLURM 环境使用 GPU

创建作业脚本 `run_scotv2_gpu.slurm`：

```bash
#!/bin/bash
#SBATCH --job-name=scotv2_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=100G

source /usr/local/opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/nfs_share/tlj/envs/scMIAC/

cd /path/to/project
bash scripts/run_methods/run/run_scotv2.sh --device cuda share
```

提交作业：
```bash
sbatch run_scotv2_gpu.slurm
```

## 算法说明

SCOTv2 是一种基于最优传输（Optimal Transport）的无监督多模态整合方法：
- 不需要细胞类型标签信息
- 基于 Gromov-Wasserstein 最优传输对齐跨模态数据
- 支持 CPU 和 GPU 计算

### GPU 加速说明

- **加速部分**：Gromov-Wasserstein 优化中的 PyTorch Tensor 运算（`torch.einsum`、Sinkhorn 迭代）
- **CPU 瓶颈**：图距离计算（`scipy.sparse.csgraph.dijkstra`）和矩阵特征分解（`np.linalg.eig`）
- **预期加速比**：1.2x - 2x（取决于数据规模和 GPU 型号）
- **向后兼容**：默认使用 CPU，不带 `--device` 参数时行为与原版本相同

### 自动降级

如果指定 `--device cuda` 但 CUDA 不可用，算法会自动降级到 CPU 并输出警告：
```
Warning: CUDA requested but not available, using CPU instead
SCOTv2 using device: cpu
```

## 依赖

- PyTorch（支持 CUDA）
- POT (Python Optimal Transport)
- numpy
- scipy
- scanpy
- anndata
- scikit-learn

## 输出文件

运行完成后，输出目录包含：
- `rna_embeddings.csv` - RNA 模态嵌入向量
- `atac_embeddings.csv` - ATAC 模态嵌入向量
- `scotv2_latent_umap.png` - UMAP 可视化图片
- `combined.h5ad` - 合并的 AnnData 对象

## 参考

- 原始实现：https://github.com/rsinghlab/SCOT
- 论文：Demetci et al. (2022) "SCOT: Single-Cell Multi-Omics Alignment with Optimal Transport" Journal of Computational Biology
