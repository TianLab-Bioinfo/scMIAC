# scMIAC 实验脚本目录

本目录包含 scMIAC 方法的实验扩展脚本，用于支持消融实验、超参数测试和 GPU 监控。

## 架构说明

### scmiac CLI vs main.py 的职责分工

- **scmiac CLI** (`scmiac/cli.py`)
  - 纯净的原版 scMIAC 方法实现
  - 仅支持 MNN 锚点生成
  - 固定使用对比学习损失
  - 始终启用细胞类型过滤
  - 适合生产使用和标准 benchmark

- **main.py** (本文件)
  - 实验研究平台
  - 支持所有消融实验功能
  - 支持超参数测试
  - GPU 监控和资源统计
  - 注释噪声注入测试
  - 适合研究实验和方法分析

## 文件说明

- `main.py`: scMIAC 实验运行脚本，直接调用 scmiac 包函数
- `utils.py`: 实验性工具函数（随机锚点、注释噪声注入等）
- `__init__.py`: 包初始化文件

## 使用方法

### 标准运行（等同于 CLI）

```bash
python main.py --dataset 10x --output-dir data/10x/output/methods/scmiac
python main.py --dataset share --output-dir data/share/output/methods/scmiac --num-epochs 50
```

### 消融实验

#### w/o MNN（使用随机锚点）
```bash
python main.py --dataset 10x --output-dir data/10x/output/ablation/wo_mnn \
    --anchor-generation random --random-anchor-seed 42
```

#### w/o Contrastive Learning（使用 MSE 损失）
```bash
python main.py --dataset 10x --output-dir data/10x/output/ablation/wo_contra \
    --anchor-loss-type mse
```

#### w/o Cell Type Filtering
```bash
python main.py --dataset 10x --output-dir data/10x/output/ablation/wo_ctf \
    --no-ct-filter
```

#### w/o VAE（关闭重构和 KL 损失）
```bash
python main.py --dataset 10x --output-dir data/10x/output/ablation/wo_vae \
    --lambda-rna-kl 0.0 --lambda-atac-kl 0.0 \
    --alpha-rna-rec 0.0 --alpha-atac-rec 0.0
```

### 超参数测试

#### 注释准确率测试
```bash
python main.py --dataset 10x --output-dir data/10x/output/hyperparameter/anno_acc/80 \
    --anno-accuracy 80 --anno-seed 42
```

#### 对比学习权重测试
```bash
python main.py --dataset 10x --output-dir data/10x/output/hyperparameter/lambda_contra/100 \
    --lambda-contra 100 --identifier "lambda_contra_100"
```

## 主要参数说明

### 必需参数
- `--dataset`: 数据集名称（或使用 --rna-h5ad-override 和 --atac-h5ad-override）
- `--output-dir`: 输出目录

### 锚点生成参数（消融实验）
- `--anchor-generation {mnn,random}`: 锚点生成策略（默认: mnn）
- `--random-anchor-seed`: 随机锚点种子
- `--random-anchor-count`: 随机锚点数量
- `--no-ct-filter`: 禁用细胞类型过滤

### 损失函数参数（消融实验）
- `--anchor-loss-type {contrastive,mse}`: 锚点对齐损失类型（默认: contrastive）
- `--lambda-rna-kl`: RNA KL 散度权重（默认: 1.0）
- `--lambda-atac-kl`: ATAC KL 散度权重（默认: 1.0）
- `--alpha-rna-rec`: RNA 重构损失权重（默认: 20.0）
- `--alpha-atac-rec`: ATAC 重构损失权重（默认: 20.0）
- `--lambda-contra`: 对比学习损失权重（默认: 200.0）

### 训练参数
- `--device`: 设备（默认: cuda:0）
- `--num-epochs`: 训练轮数（默认: 2000）
- `--batch-size`: 批量大小（默认: 1024）
- `--learning-rate`: 学习率（默认: 1e-3）
- `--print-step`: 打印间隔（默认: 10）

### 实验性功能
- `--anno-accuracy [0-100]`: 注释准确率（用于测试鲁棒性）
- `--anno-seed`: 注释噪声种子（默认: 42）
- `--identifier`: 实验标识符（用于 GPU CSV 记录）
- `--gpu-csv`: 自定义 GPU 统计文件路径

## 输出文件

标准输出：
- `anchors.csv`: 锚点细胞对
- `rna_vae.pth`: RNA VAE 模型权重
- `atac_vae.pth`: ATAC VAE 模型权重
- `rna_embeddings.csv`: RNA 细胞嵌入
- `atac_embeddings.csv`: ATAC 细胞嵌入
- `scmiac_latent_umap.png`: UMAP 可视化

GPU 监控输出（保存在上级目录）：
- `../gpu.csv`: GPU 使用统计

```csv
method,gpu_id,peak_memory_mb,peak_memory_gb,avg_utilization_percent,max_utilization_percent,sampling_count,timestamp
scmiac,0,8432.5,8.23,87.3,99.8,200,2025-10-23 10:30:15
wo_mnn,0,8521.2,8.32,89.1,100.0,205,2025-10-23 11:15:30
```

## 与 CLI 的对比

| 功能 | scmiac CLI | main.py |
|------|-----------|---------|
| MNN 锚点 | ✅ | ✅ |
| 随机锚点 | ❌ | ✅ |
| 对比学习损失 | ✅ 固定 | ✅ |
| MSE 损失 | ❌ | ✅ |
| 细胞类型过滤 | ✅ 固定启用 | ✅ 可选 |
| GPU 监控 | ❌ | ✅ |
| 注释噪声注入 | ❌ | ✅ |
| 任意权重组合 | ❌ | ✅ |
| 使用场景 | 生产/Benchmark | 研究/实验 |

## 调用关系

```
scripts/
├── run_methods/run/run_scmiac.sh          → main.py (标准 benchmark)
├── ablation/wo_*.sh                        → main.py (消融实验)
└── hyperparameter/*/*.sh                   → main.py (超参数测试)
```

所有脚本都通过 main.py 统一入口，不再直接调用 scmiac CLI。
