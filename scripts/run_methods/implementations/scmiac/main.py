#!/usr/bin/env python
"""
scMIAC 方法实验运行脚本（带 GPU 监控和实验性功能）

这是 scMIAC 核心包的扩展脚本，用于研究实验和超参数测试。
直接调用 scmiac 包中的函数，支持所有实验性参数和消融实验。

与 scmiac CLI 的区别：
- CLI: 纯净的原版 scMIAC 方法（仅 MNN + Contrastive Learning）
- main.py: 支持消融实验、超参数测试、注释噪声、GPU 监控等

Usage:
    # 标准运行（等同于 CLI）
    python main.py --dataset 10x --output-dir data/10x/output/methods/scmiac

    # 消融实验：w/o MNN（使用随机锚点）
    python main.py --dataset 10x --output-dir data/10x/output/ablation/wo_mnn \\
        --anchor-generation random --random-anchor-seed 42

    # 消融实验：w/o Contrastive（使用 MSE 损失）
    python main.py --dataset 10x --output-dir data/10x/output/ablation/wo_contra \\
        --anchor-loss-type mse

    # 消融实验：w/o Cell Type Filtering
    python main.py --dataset 10x --output-dir data/10x/output/ablation/wo_ctf \\
        --no-ct-filter

    # 消融实验：w/o VAE（关闭重构和 KL 损失）
    python main.py --dataset 10x --output-dir data/10x/output/ablation/wo_vae \\
        --lambda-rna-kl 0.0 --lambda-atac-kl 0.0 --alpha-rna-rec 0.0 --alpha-atac-rec 0.0

    # 超参数测试：注释准确率
    python main.py --dataset 10x --output-dir data/10x/output/hyperparameter/anno_acc/80 \\
        --anno-accuracy 80 --anno-seed 42

    # 超参数测试：对比学习权重
    python main.py --dataset 10x --output-dir data/10x/output/hyperparameter/lambda_contra/100 \\
        --lambda-contra 100
"""

import argparse
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F

# 添加父目录到路径以导入 utils 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入 scmiac 核心包
from scmiac.modeling.scmiac import find_anchors, preprocess, train_model, model_inference
from scmiac.preprocessing.preprocess import run_umap
from scmiac.utils import set_seed

# 导入实验性工具函数
try:
    from utils.gpu_monitor import GPUMonitor, save_gpu_stats, PYNVML_AVAILABLE
except ImportError as e:
    print(f"Warning: Failed to import GPU monitor utilities: {e}", file=sys.stderr)
    print("GPU monitoring will be disabled", file=sys.stderr)
    PYNVML_AVAILABLE = False

# 导入本地工具函数
from implementations.scmiac.utils import generate_random_anchors, inject_annotation_noise, augment_anchors_with_paired_cells


def run_scmiac_with_experiments(
    adata_rna,
    adata_atac,
    output_dir,
    # 锚点生成参数
    anchor_csv_path=None,
    anchor_generation='mnn',
    all_nfeatures=3000,
    single_nfeatures=2000,
    k_anchor=5,
    n_components=30,
    ct_filter=True,
    mode=None,
    random_anchor_seed=None,
    random_anchor_count=None,
    rna_celltype_key="cell_type",
    atac_celltype_key="pred",
    paired_cells_file=None,
    # 数据预处理参数
    rna_latent_key="X_pca",
    atac_latent_key="lsi49",
    batch_size=256,
    hidden_dims=None,
    latent_dim=30,
    balanced_sampler=True,
    # 训练参数
    device='cuda:0',
    num_epochs=2000,
    lambda_rna_kl=1.0,
    lambda_atac_kl=1.0,
    alpha_rna_rec=20.0,
    alpha_atac_rec=20.0,
    lambda_contra=300.0,
    temperature=0.5,
    anchor_loss_type='contrastive',
    learning_rate=1e-3,
    print_step=10,
    # 输出参数
    plot_umap=True,
    umap_dpi=150,
):
    """
    运行 scMIAC 训练（支持所有实验性功能）
    
    Args:
        adata_rna: RNA AnnData 对象
        adata_atac: ATAC AnnData 对象
        output_dir: 输出目录 Path 对象
        anchor_csv_path: 锚点 CSV 文件路径（可选，不存在则生成）
        anchor_generation: 锚点生成策略 ('mnn', 'random')
        ... （其他参数见函数签名）
    
    Returns:
        字典，包含训练好的模型和嵌入
    """
    if hidden_dims is None:
        hidden_dims = [128, 64]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 步骤 1: 生成锚点（总是重新生成，不读取已存在的）
    if anchor_csv_path is None:
        anchor_csv_path = output_dir / 'anchors.csv'
    else:
        anchor_csv_path = Path(anchor_csv_path)
    
    print(f"Generating anchors (will overwrite if exists)...")
    
    if anchor_generation == 'mnn':
        # 使用 MNN 生成锚点
        print("Using MNN anchor generation strategy")
        anchor_df = find_anchors(
            adata_rna,
            adata_atac,
            all_nfeatures=all_nfeatures,
            single_nfeatures=single_nfeatures,
            k_anchor=k_anchor,
            n_components=n_components,
            ct_filter=ct_filter,
            mode=mode,
            rna_celltype_key=rna_celltype_key,
            atac_celltype_key=atac_celltype_key,
        )
        print(f"MNN anchors generated: {len(anchor_df)} pairs")
        
    elif anchor_generation == 'random':
        # 使用随机锚点（消融实验）
        print("Using random anchor generation strategy (ablation: w/o MNN)")
        
        # 确定基线锚点数量
        if random_anchor_count is not None:
            baseline_count = random_anchor_count
            print(f"Using user-specified count: {baseline_count}")
        else:
            print("Generating baseline MNN anchors to determine count...")
            baseline_df = find_anchors(
                adata_rna,
                adata_atac,
                all_nfeatures=all_nfeatures,
                single_nfeatures=single_nfeatures,
                k_anchor=k_anchor,
                n_components=n_components,
                ct_filter=ct_filter,
                mode=mode,
                rna_celltype_key=rna_celltype_key,
                atac_celltype_key=atac_celltype_key,
            )
            baseline_count = len(baseline_df)
            print(f"Baseline MNN anchor count: {baseline_count}")
        
        if baseline_count == 0:
            raise ValueError("Baseline anchor count is zero; cannot generate random anchors")
        
        # 生成随机锚点
        if random_anchor_seed is None:
            random_anchor_seed = 42
        
        anchor_df = generate_random_anchors(
            adata_rna,
            adata_atac,
            count=baseline_count,
            seed=random_anchor_seed,
            rna_celltype_key=rna_celltype_key,
            atac_celltype_key=atac_celltype_key,
        )
        print(f"Random anchors generated: {len(anchor_df)} pairs (seed={random_anchor_seed})")
    else:
        raise ValueError(f"Unknown anchor generation strategy: {anchor_generation}")
    
    # 保存锚点（覆盖已存在的文件）
    anchor_csv_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_df.to_csv(anchor_csv_path, index=False)
    print(f"Anchors saved to {anchor_csv_path}")
    
    # 步骤 2: 数据预处理
    print("\n" + "=" * 60)
    print("Preprocessing data...")
    print("=" * 60)
    rna_vae, atac_vae, all_cells_loader, anchor_cells_loader = preprocess(
        adata_rna,
        adata_atac,
        anchor_df,
        rna_latent_key=rna_latent_key,
        atac_latent_key=atac_latent_key,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        balanced_sampler=balanced_sampler,
        device=device,
    )
    
    # 步骤 3: 训练模型
    print("\n" + "=" * 60)
    print("Training VAE models...")
    print(f"Device: {device}")
    print(f"Total epochs: {num_epochs}")
    print(f"Anchor loss type: {anchor_loss_type}")
    print(f"Loss weights: λ_RNA_KL={lambda_rna_kl}, λ_ATAC_KL={lambda_atac_kl}, "
          f"α_RNA_rec={alpha_rna_rec}, α_ATAC_rec={alpha_atac_rec}, λ_contra={lambda_contra}")
    print("=" * 60)
    
    rna_vae, atac_vae = train_model(
        rna_vae,
        atac_vae,
        all_cells_loader,
        anchor_cells_loader,
        device=device,
        num_epoches=num_epochs,
        lambda_rna_kl=lambda_rna_kl,
        lambda_atac_kl=lambda_atac_kl,
        alpha_rna_rec=alpha_rna_rec,
        alpha_atac_rec=alpha_atac_rec,
        lambda_contra=lambda_contra,
        temperature=temperature,
        anchor_loss_type=anchor_loss_type,
        lr=learning_rate,
        print_step=print_step,
        save_model=False,
    )
    
    # 步骤 4: 保存模型权重
    print("\n" + "=" * 60)
    print("Saving model weights...")
    print("=" * 60)
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    rna_vae_path = output_dir / 'rna_vae.pth'
    atac_vae_path = output_dir / 'atac_vae.pth'
    torch.save(rna_vae.state_dict(), rna_vae_path)
    torch.save(atac_vae.state_dict(), atac_vae_path)
    print(f"RNA VAE saved to: {rna_vae_path}")
    print(f"ATAC VAE saved to: {atac_vae_path}")
    
    # 步骤 5: 生成嵌入
    print("\n" + "=" * 60)
    print("Generating embeddings...")
    print("=" * 60)
    rna_embeddings, atac_embeddings = model_inference(
        rna_vae,
        atac_vae,
        all_cells_loader,
        device=device,
    )
    
    rna_embeddings_path = output_dir / 'rna_embeddings.csv'
    atac_embeddings_path = output_dir / 'atac_embeddings.csv'
    
    # 直接保存所有embeddings（如果提供了配对文件，adata已在main()中过滤）
    pd.DataFrame(rna_embeddings, index=adata_rna.obs_names).to_csv(rna_embeddings_path)
    pd.DataFrame(atac_embeddings, index=adata_atac.obs_names).to_csv(atac_embeddings_path)
    
    print(f"RNA embeddings saved to: {rna_embeddings_path} ({len(adata_rna)} cells)")
    print(f"ATAC embeddings saved to: {atac_embeddings_path} ({len(adata_atac)} cells)")
    
    # 步骤 6: UMAP 可视化
    if plot_umap:
        print("\n" + "=" * 60)
        print("Generating UMAP visualization...")
        print("=" * 60)
        
        def prepare_adata(source, embeddings, modality):
            prepared = source.copy()
            prepared.obsm["scmiac_latent"] = embeddings
            if "modality" not in prepared.obs:
                prepared.obs["modality"] = modality
            else:
                prepared.obs["modality"] = prepared.obs["modality"].astype(str).fillna(modality)
            if "cell_type" not in prepared.obs:
                prepared.obs["cell_type"] = f"unknown_{modality.lower()}"
            else:
                prepared.obs["cell_type"] = prepared.obs["cell_type"].astype(str).fillna(
                    f"unknown_{modality.lower()}"
                )
            return prepared
        
        prepared_rna = prepare_adata(adata_rna, rna_embeddings, "RNA")
        prepared_atac = prepare_adata(adata_atac, atac_embeddings, "ATAC")
        
        adata_cm = ad.concat([prepared_rna, prepared_atac], join="outer", index_unique=None)
        adata_cm = run_umap(adata_cm, "scmiac_latent")
        
        umap_output = output_dir / 'scmiac_latent_umap.png'
        fig = sc.pl.embedding(
            adata_cm,
            basis="scmiac_latent_umap",
            color=["cell_type", "modality"],
            legend_loc="on data",
            show=False,
            return_fig=True,
        )
        fig.savefig(umap_output, bbox_inches="tight", dpi=umap_dpi)
        print(f"UMAP plot saved to: {umap_output}")
    
    print("\n" + "=" * 60)
    print("scMIAC training completed successfully!")
    print("=" * 60)
    
    return {
        'rna_vae': rna_vae,
        'atac_vae': atac_vae,
        'rna_embeddings': rna_embeddings,
        'atac_embeddings': atac_embeddings,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run scMIAC method with experimental features and GPU monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('--dataset', help='Dataset name (required unless --rna-h5ad and --atac-h5ad are provided)')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--data-root', default='data', help='Data root directory')
    
    # 可选：直接指定输入文件路径
    parser.add_argument('--rna-h5ad-override', help='Override RNA h5ad file path')
    parser.add_argument('--atac-h5ad-override', help='Override ATAC h5ad file path')
    
    # 锚点生成参数
    parser.add_argument('--anchors-csv', help='Path to anchors CSV file (optional, will be generated if not exists)')
    parser.add_argument('--anchor-generation', type=str, choices=['mnn', 'random'],
                       default='mnn', help='Anchor generation strategy (default: mnn)')
    parser.add_argument('--all-nfeatures', type=int, default=3000,
                       help='Number of features for integration when generating anchors')
    parser.add_argument('--single-nfeatures', type=int, default=2000,
                       help='Number of features per modality when generating anchors')
    parser.add_argument('--k-anchor', type=int, default=5,
                       help='Number of neighbors when selecting anchors')
    parser.add_argument('--n-components', type=int, default=30,
                       help='Dimensionality for CCA during anchor finding')
    parser.add_argument('--no-ct-filter', action='store_true',
                       help='Disable cell type filtering when generating anchors (for ablation study)')
    parser.add_argument('--mode', choices=['v'], default=None,
                       help='Set to "v" for vertical integration (paired data)')
    parser.add_argument('--random-anchor-seed', type=int,
                       help='Random seed for anchor generation when --anchor-generation random')
    parser.add_argument('--random-anchor-count', type=int,
                       help='Number of random anchors to generate (defaults to MNN baseline count)')
    parser.add_argument('--rna-celltype-key', default='cell_type',
                       help='Key for RNA cell type annotation in AnnData.obs')
    parser.add_argument('--atac-celltype-key', default='pred',
                       help='Key for ATAC cell type annotation in AnnData.obs')
    parser.add_argument('--paired-cells-file', type=str,
                       help='Path to paired cells file for vertical integration (one cell ID per line)')
    
    # 数据预处理参数
    parser.add_argument('--rna-latent-key', default='X_pca',
                       help='Key for RNA latent representation in AnnData')
    parser.add_argument('--atac-latent-key', default='lsi49',
                       help='Key for ATAC latent representation in AnnData')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--hidden-dims', type=str, default='128,64',
                       help='Comma-separated hidden dimensions for VAE')
    parser.add_argument('--latent-dim', type=int, default=30,
                       help='Latent dimension size for VAE')
    parser.add_argument('--no-balanced-sampler', action='store_true',
                       help='Disable balanced anchor sampling')
    
    # 训练参数
    parser.add_argument('--device', default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--num-epochs', type=int, default=2000, help='Number of training epochs')

    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--print-step', type=int, default=10, help='Print loss every N epochs')
    
    # 损失函数权重参数
    parser.add_argument('--lambda-rna-kl', type=float, default=1.0,
                       help='Weight for RNA KL divergence loss')
    parser.add_argument('--lambda-atac-kl', type=float, default=1.0,
                       help='Weight for ATAC KL divergence loss')
    parser.add_argument('--alpha-rna-rec', type=float, default=20.0,
                       help='Weight for RNA reconstruction loss')
    parser.add_argument('--alpha-atac-rec', type=float, default=20.0,
                       help='Weight for ATAC reconstruction loss')
    parser.add_argument('--lambda-contra', type=float, default=300.0,
                       help='Weight for anchor alignment loss')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for contrastive loss')
    parser.add_argument('--anchor-loss-type', type=str, choices=['contrastive', 'mse'],
                       default='contrastive',
                       help='Anchor loss type: contrastive or mse (for ablation study)')
    
    # 输出参数
    parser.add_argument('--no-plot-umap', action='store_true',
                       help='Disable UMAP visualization')
    parser.add_argument('--umap-dpi', type=int, default=150,
                       help='DPI for UMAP plot')
    
    # 实验性功能参数
    parser.add_argument('--gpu-csv', type=str,
                       help='Custom path for GPU stats CSV file')
    parser.add_argument('--identifier', type=str,
                       help='Identifier for GPU/stats CSV (default: scmiac)')
    parser.add_argument('--anno-accuracy', type=int, choices=range(0, 101), metavar='[0-100]',
                       help='Annotation accuracy (0-100). Randomly replace (100-accuracy)%% labels with other cell types')
    parser.add_argument('--anno-seed', type=int, default=42,
                       help='Random seed for annotation noise injection')
    
    args = parser.parse_args()
    
    # 确保使用绝对路径，避免相对路径问题
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建输入文件路径
    if args.rna_h5ad_override and args.atac_h5ad_override:
        rna_h5ad = Path(args.rna_h5ad_override)
        atac_h5ad = Path(args.atac_h5ad_override)
        dataset = args.dataset if args.dataset else "custom"
    elif args.dataset:
        data_root = Path(args.data_root)
        dataset = args.dataset
        rna_h5ad = data_root / dataset / "input" / f"adata_rna_{dataset}.h5ad"
        atac_h5ad = data_root / dataset / "input" / f"adata_atac_{dataset}.h5ad"
    else:
        print("Error: Either --dataset or both --rna-h5ad-override and --atac-h5ad-override must be provided",
              file=sys.stderr)
        sys.exit(1)
    
    if not rna_h5ad.exists():
        print(f"Error: RNA data not found: {rna_h5ad}", file=sys.stderr)
        sys.exit(1)
    if not atac_h5ad.exists():
        print(f"Error: ATAC data not found: {atac_h5ad}", file=sys.stderr)
        sys.exit(1)
    
    # 设置随机种子（确保可重复性，匹配原始scMIAAC实现）
    set_seed(seed=24)
    
    # 加载数据
    print("=" * 60)
    print(f"Loading data from dataset: {dataset}")
    print(f"RNA: {rna_h5ad}")
    print(f"ATAC: {atac_h5ad}")
    print("=" * 60)
    adata_rna = sc.read_h5ad(rna_h5ad)
    adata_atac = sc.read_h5ad(atac_h5ad)
    
    # 注入注释噪声（如果指定了 anno-accuracy）
    temp_files = []
    if args.anno_accuracy is not None:
        print("\n" + "=" * 60)
        print(f"Injecting annotation noise: {args.anno_accuracy}% accuracy")
        print(f"Random seed: {args.anno_seed}")
        print("=" * 60)
        
        temp_rna = output_dir / f"_temp_rna_acc{args.anno_accuracy}.h5ad"
        temp_atac = output_dir / f"_temp_atac_acc{args.anno_accuracy}.h5ad"
        
        # 只对 ATAC 注入噪声，RNA 保持原标签（模拟真实场景：RNA注释准确，ATAC注释有误差）
        print("\nProcessing ATAC data (injecting annotation noise):")
        inject_annotation_noise(
            atac_h5ad, args.anno_accuracy, temp_atac,
            seed=args.anno_seed, celltype_key=args.atac_celltype_key
        )
        
        # 重新加载注入噪声后的 ATAC 数据，RNA 保持不变
        adata_atac = sc.read_h5ad(temp_atac)
        temp_files = [temp_atac]
        
        print("\n" + "=" * 60)
        print("Annotation noise injection completed")
        print("=" * 60 + "\n")
    
    # 如果提供了配对细胞文件，过滤掉配对细胞（只在未配对细胞上运行）
    if args.paired_cells_file:
        paired_cells_path = Path(args.paired_cells_file)
        if paired_cells_path.is_file():
            print("\n" + "=" * 60)
            print("Filtering out paired cells from datasets")
            print("=" * 60)
            
            # 读取配对细胞列表
            with open(paired_cells_path, 'r') as f:
                paired_cells = set(line.strip() for line in f if line.strip())
            print(f"Paired cells to exclude: {len(paired_cells)}")
            
            # 统计原始细胞数
            n_rna_orig = adata_rna.n_obs
            n_atac_orig = adata_atac.n_obs
            
            # 过滤未配对细胞
            unpaired_rna_mask = [c not in paired_cells for c in adata_rna.obs_names]
            unpaired_atac_mask = [c not in paired_cells for c in adata_atac.obs_names]
            
            adata_rna = adata_rna[unpaired_rna_mask, :].copy()
            adata_atac = adata_atac[unpaired_atac_mask, :].copy()
            
            n_rna_unpaired = adata_rna.n_obs
            n_atac_unpaired = adata_atac.n_obs
            
            print(f"RNA cells: {n_rna_orig} -> {n_rna_unpaired} (removed {n_rna_orig - n_rna_unpaired})")
            print(f"ATAC cells: {n_atac_orig} -> {n_atac_unpaired} (removed {n_atac_orig - n_atac_unpaired})")
            print("=" * 60 + "\n")
        else:
            print(f"Warning: Paired cells file not found: {paired_cells_path}", file=sys.stderr)
    
    # 解析 hidden_dims
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(',')]
    
    # 启动 GPU 监控
    gpu_monitor = None
    gpu_id = None
    if args.device.startswith('cuda') and PYNVML_AVAILABLE:
        try:
            gpu_id = int(args.device.split(':')[-1]) if ':' in args.device else 0
            gpu_monitor = GPUMonitor(gpu_id=gpu_id, sampling_interval=1.0)
            gpu_monitor.start()
            print(f"GPU monitoring started for GPU {gpu_id}")
        except Exception as e:
            print(f"Warning: Failed to start GPU monitoring: {e}")
            gpu_monitor = None
    
    exit_code = 0
    try:
        # 运行 scMIAC 训练
        results = run_scmiac_with_experiments(
            adata_rna=adata_rna,
            adata_atac=adata_atac,
            output_dir=output_dir,
            # 锚点生成参数
            anchor_csv_path=args.anchors_csv,
            anchor_generation=args.anchor_generation,
            all_nfeatures=args.all_nfeatures,
            single_nfeatures=args.single_nfeatures,
            k_anchor=args.k_anchor,
            n_components=args.n_components,
            ct_filter=not args.no_ct_filter,
            mode=args.mode,
            random_anchor_seed=args.random_anchor_seed,
            random_anchor_count=args.random_anchor_count,
            rna_celltype_key=args.rna_celltype_key,
            atac_celltype_key=args.atac_celltype_key,
            paired_cells_file=args.paired_cells_file,
            # 数据预处理参数
            rna_latent_key=args.rna_latent_key,
            atac_latent_key=args.atac_latent_key,
            batch_size=args.batch_size,
            hidden_dims=hidden_dims,
            latent_dim=args.latent_dim,
            balanced_sampler=not args.no_balanced_sampler,
            # 训练参数
            device=args.device,
            num_epochs=args.num_epochs,

            lambda_rna_kl=args.lambda_rna_kl,
            lambda_atac_kl=args.lambda_atac_kl,
            alpha_rna_rec=args.alpha_rna_rec,
            alpha_atac_rec=args.alpha_atac_rec,
            lambda_contra=args.lambda_contra,
            temperature=args.temperature,
            anchor_loss_type=args.anchor_loss_type,
            learning_rate=args.learning_rate,
            print_step=args.print_step,
            # 输出参数
            plot_umap=not args.no_plot_umap,
            umap_dpi=args.umap_dpi,
        )
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        exit_code = 130
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
        
    finally:
        # 停止 GPU 监控并保存统计数据
        if gpu_monitor:
            try:
                stats = gpu_monitor.stop()
                
                methods_dir = output_dir.parent
                csv_path = Path(args.gpu_csv) if args.gpu_csv else None
                identifier = args.identifier if args.identifier else 'scmiac'
                save_gpu_stats(
                    stats,
                    methods_dir,
                    method_name=identifier,
                    gpu_id=gpu_id,
                    csv_path=csv_path
                )
            except Exception as e:
                print(f"Warning: Failed to save GPU stats: {e}")
        
        # 清理临时文件
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    print(f"Cleaned up temporary file: {temp_file.name}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {temp_file}: {e}")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
