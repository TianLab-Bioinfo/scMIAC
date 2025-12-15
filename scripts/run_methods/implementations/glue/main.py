#!/usr/bin/env python
# GLUE 方法运行脚本
"""
运行 GLUE 方法进行多模态整合

参考文档:
- https://scglue.readthedocs.io/zh-cn/latest/preprocessing.html
- https://scglue.readthedocs.io/zh-cn/latest/training.html

Usage:
    python main.py --dataset 10x --output-dir data/10x/output/methods/glue
    python main.py --dataset share --output-dir data/share/output/methods/glue
"""

import argparse
import os
import sys
from pathlib import Path

import anndata
import itertools
import networkx as nx
import scanpy as sc
import pandas as pd
import numpy as np
import scglue

# 添加父目录到路径以导入utils模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from utils.gpu_monitor import GPUMonitor, save_gpu_stats, PYNVML_AVAILABLE
except ImportError as e:
    print(f"Warning: Failed to import GPU monitor: {e}", file=sys.stderr)
    PYNVML_AVAILABLE = False
    GPUMonitor = None
    save_gpu_stats = None


def get_genome_file(dataset_name: str, genome_dir: str) -> Path:
    """根据数据集获取基因组注释文件路径"""
    genome_dir = Path(genome_dir)
    
    # share 和 LungDroplet 是小鼠数据，其他是人类数据
    if dataset_name.lower() in ['share', 'lungdroplet']:
        genome_file = genome_dir / "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz"
        species = "mouse"
    else:
        genome_file = genome_dir / "gencode.v40.chr_patch_hapl_scaff.annotation.gtf.gz"
        species = "human"
    
    if not genome_file.exists():
        raise FileNotFoundError(f"Genome annotation file not found: {genome_file}")
    
    print(f"Using {species} genome: {genome_file}")
    return genome_file


def load_raw_data(dataset_name: str, data_root: str = "data"):
    """加载原始未处理的 RNA 和 ATAC 数据（含原始整数计数）"""
    print(f"Loading raw {dataset_name} dataset...")
    
    data_dir = Path(data_root) / dataset_name / "input"
    
    # 使用包含原始整数计数的文件
    rna_path = data_dir / f"adata_rnacounts_{dataset_name}.h5ad"
    atac_path = data_dir / f"adata_peak_{dataset_name}.h5ad"
    
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA data not found: {rna_path}")
    if not atac_path.exists():
        raise FileNotFoundError(f"ATAC data not found: {atac_path}")
    
    rna = anndata.read_h5ad(rna_path)
    atac = anndata.read_h5ad(atac_path)
    
    print(f"RNA shape: {rna.shape}")
    print(f"ATAC shape: {atac.shape}")
    return rna, atac


def preprocess_rna(rna, genome_file: Path):
    """预处理 RNA 数据：备份原始计数、标记高变基因、添加基因组位置信息"""
    print("Preprocessing RNA data...")
    
    # 1. 备份原始计数
    if 'counts' not in rna.layers:
        print("  Backing up raw counts to layers['counts']")
        rna.layers['counts'] = rna.X.copy()
    
    # 2. 计算 highly_variable（必须在构建图之前完成）
    if 'highly_variable' not in rna.var.columns:
        print("  Computing highly variable genes...")
        sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor='seurat_v3')
    else:
        print(f"  Using existing highly_variable annotation ({rna.var['highly_variable'].sum()} genes)")
    
    # 3. 使用 scglue 添加基因组坐标
    print("  Adding genomic coordinates...")
    scglue.data.get_gene_annotation(
        rna, 
        gtf=str(genome_file),
        gtf_by="gene_name"
    )
    
    # 4. 过滤掉没有基因组位置信息的基因
    gene_index = np.where(np.isfinite(rna.var['chromStart'].values))[0]
    print(f"  Keeping {len(gene_index)}/{rna.shape[1]} genes with valid genomic coordinates")
    rna = rna[:, gene_index]
    
    return rna


def preprocess_atac(atac):
    """预处理 ATAC 数据：备份原始计数、标记高变 peaks、提取染色体位置"""
    print("Preprocessing ATAC data...")
    
    # 1. 备份原始计数
    if 'counts' not in atac.layers:
        print("  Backing up raw counts to layers['counts']")
        atac.layers['counts'] = atac.X.copy()
    
    # 2. 计算 highly_variable peaks（必须在构建图之前完成）
    if 'highly_variable' not in atac.var.columns:
        print("  Computing highly variable peaks...")
        # ATAC 数据通常很稀疏，使用 seurat 方法
        try:
            sc.pp.highly_variable_genes(atac, n_top_genes=30000, flavor='seurat')
        except Exception as e:
            print(f"  WARNING: Failed to compute highly_variable: {e}")
            print(f"  Marking all peaks as highly variable")
            atac.var['highly_variable'] = True
    else:
        print(f"  Using existing highly_variable annotation ({atac.var['highly_variable'].sum()} peaks)")
    
    # 3. ATAC peaks 格式通常是 "chr:start-end" 或 "chr-start-end"
    print("  Extracting genomic coordinates...")
    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.map(lambda x: x[0])
    atac.var["chromStart"] = split.map(lambda x: int(x[1]))
    atac.var["chromEnd"] = split.map(lambda x: int(x[2]))
    
    print(f"  Extracted genomic coordinates for {atac.shape[1]} peaks")
    return atac


def build_prior_graph(rna, atac, output_dir: Path):
    """构建先验调控图"""
    print("Building prior regulatory graph...")
    
    # 使用 scglue 的 rna_anchored_guidance_graph 构建调控图
    # 这会基于基因的 upstream 区域与 peaks 的重叠来构建图
    graph = scglue.genomics.rna_anchored_guidance_graph(
        rna, atac,
        extend_range=150000  # 基因上游 150kb 范围
    )
    
    print(f"Graph created: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # 保存图
    graph_file = output_dir / "prior.graphml.gz"
    nx.write_graphml(graph, graph_file)
    print(f"Graph saved to {graph_file}")
    
    return graph


def prepare_for_training(rna, atac):
    """准备训练数据：标准化和降维"""
    print("Preparing data for training...")
    
    # RNA 数据准备
    print("  Processing RNA...")
    
    # 检查是否已有原始计数层
    if 'counts' not in rna.layers:
        # 尝试从 raw 获取原始计数
        if rna.raw is not None:
            print("    Using rna.raw.X as counts layer")
            # 只保留当前筛选的基因
            rna.layers['counts'] = rna.raw[:, rna.var_names].X
        else:
            print("    WARNING: No raw counts available, using current rna.X")
            rna.layers['counts'] = rna.X.copy()
    
    # 检查是否已有 highly_variable 标记
    if 'highly_variable' not in rna.var.columns:
        print("    Computing highly variable genes...")
        sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor='seurat_v3')
    else:
        print(f"    Using existing highly_variable annotation ({rna.var['highly_variable'].sum()} genes)")
    
    # 检查是否已有 PCA
    if 'X_pca' not in rna.obsm:
        print("    Computing PCA...")
        # 如果数据未标准化，先标准化
        if rna.X.mean() > 1 or rna.X.min() < 0:
            sc.pp.normalize_total(rna)
            sc.pp.log1p(rna)
            sc.pp.scale(rna)
        sc.tl.pca(rna, n_comps=100, svd_solver='auto')
    else:
        print(f"    Using existing PCA (shape: {rna.obsm['X_pca'].shape})")
    
    # ATAC 数据准备
    print("  Processing ATAC...")
    
    # 备份原始计数
    if 'counts' not in atac.layers:
        atac.layers['counts'] = atac.X.copy()
    
    # 检查是否已有 highly_variable 标记
    if 'highly_variable' not in atac.var.columns:
        print("    Computing highly variable peaks...")
        # ATAC 数据通常很稀疏，使用 seurat 方法而不是 seurat_v3
        try:
            sc.pp.highly_variable_genes(atac, n_top_genes=30000, flavor='seurat')
        except Exception as e:
            print(f"    WARNING: Failed to compute highly_variable: {e}")
            print(f"    Marking all peaks as highly variable")
            atac.var['highly_variable'] = True
    else:
        print(f"    Using existing highly_variable annotation ({atac.var['highly_variable'].sum()} peaks)")
    
    # ATAC 使用 LSI (TF-IDF + SVD)
    if 'X_lsi' not in atac.obsm:
        print("    Computing LSI...")
        scglue.data.lsi(atac, n_components=100, n_iter=15)
    else:
        print(f"    Using existing LSI (shape: {atac.obsm['X_lsi'].shape})")
    
    print(f"RNA PCA shape: {rna.obsm['X_pca'].shape}")
    print(f"ATAC LSI shape: {atac.obsm['X_lsi'].shape}")
    
    return rna, atac


def configure_datasets(rna, atac, graph):
    """配置数据集用于 GLUE 模型"""
    print("Configuring datasets for GLUE...")
    
    scglue.models.configure_dataset(
        rna, "NB", 
        use_highly_variable=True,
        use_layer="counts", 
        use_rep="X_pca"
    )
    
    scglue.models.configure_dataset(
        atac, "NB", 
        use_highly_variable=True,
        use_rep="X_lsi"
    )
    
    # 将图限制在高变基因/峰值上
    hvg_rna = rna.var.query("highly_variable").index
    hvg_atac = atac.var.query("highly_variable").index
    
    print(f"HVG RNA: {len(hvg_rna)}, HVG ATAC: {len(hvg_atac)}")
    
    graph = graph.subgraph(itertools.chain(hvg_rna, hvg_atac))
    
    print(f"Filtered graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return rna, atac, graph


def train_or_load_glue(rna, atac, graph, output_dir: Path, train: bool = False):
    """训练或加载 GLUE 模型"""
    model_path = output_dir / "glue.dill"
    
    if train or not model_path.exists():
        print("Training GLUE model...")
        print("This may take a while...")
        
        glue = scglue.models.fit_SCGLUE(
            {"rna": rna, "atac": atac}, 
            graph,
            fit_kws={"directory": str(output_dir)}
        )
        glue.save(str(model_path))
        print(f"Model saved to {model_path}")
    else:
        print(f"Loading pre-trained model from {model_path}")
        glue = scglue.models.load_model(str(model_path))
    
    return glue


def encode_and_save(glue, rna, atac, output_dir: Path):
    """使用 GLUE 模型编码数据并保存 embeddings"""
    print("Encoding data with GLUE...")
    
    rna_embeddings = glue.encode_data("rna", rna)
    atac_embeddings = glue.encode_data("atac", atac)
    
    print(f"RNA embeddings shape: {rna_embeddings.shape}")
    print(f"ATAC embeddings shape: {atac_embeddings.shape}")
    
    # 保存为 CSV 格式
    rna_embeddings_csv = output_dir / "rna_embeddings.csv"
    atac_embeddings_csv = output_dir / "atac_embeddings.csv"
    
    pd.DataFrame(rna_embeddings, index=rna.obs_names).to_csv(rna_embeddings_csv)
    pd.DataFrame(atac_embeddings, index=atac.obs_names).to_csv(atac_embeddings_csv)
    
    print(f"RNA embeddings saved to {rna_embeddings_csv}")
    print(f"ATAC embeddings saved to {atac_embeddings_csv}")
    
    return rna_embeddings, atac_embeddings


def generate_umap(rna, atac, rna_embeddings, atac_embeddings, output_dir: Path):
    """生成 UMAP 可视化"""
    print("Generating UMAP visualization...")
    
    # 添加 embeddings 到 obsm
    rna.obsm["X_glue"] = rna_embeddings
    atac.obsm["X_glue"] = atac_embeddings
    
    # 添加模态标签
    rna.obs['modality'] = "RNA"
    atac.obs['modality'] = "ATAC"
    
    # 合并数据
    combined = anndata.concat([rna, atac], join='outer')
    
    # 计算 UMAP
    sc.pp.neighbors(combined, use_rep='X_glue', key_added='glue')
    sc.tl.umap(combined, neighbors_key='glue')
    combined.obsm['X_glue_umap'] = combined.obsm['X_umap']
    
    # 保存 UMAP 图片
    umap_fig_path = output_dir / "glue_latent_umap.png"
    color_list = ["cell_type", "modality"] if "cell_type" in combined.obs else ["modality"]
    fig = sc.pl.embedding(
        combined,
        basis='X_umap',
        color=color_list,
        legend_loc="on data",
        show=False,
        return_fig=True
    )
    fig.savefig(umap_fig_path, bbox_inches='tight', dpi=150)
    print(f"UMAP plot saved to {umap_fig_path}")
    
    # 保存合并的数据
    combined_path = output_dir / "combined.h5ad"
    combined.write_h5ad(combined_path)
    print(f"Combined data saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run GLUE method for multi-modal integration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (e.g., 10x, share, wilk, zhu)'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--data-root',
        default='data',
        help='Root directory for data (default: data)'
    )
    parser.add_argument(
        '--genome-dir',
        default='data/utils/glue_genome_files',
        help='Directory containing genome annotation files (default: data/utils/glue_genome_files)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train new model instead of loading existing one'
    )
    parser.add_argument(
        '--skip-graph',
        action='store_true',
        help='Skip graph construction if prior.graphml.gz already exists'
    )
    parser.add_argument(
        '--gpu-csv',
        type=str,
        help='Custom path for GPU stats CSV file (default: data/<dataset>/output/methods/gpu.csv)'
    )
    
    args = parser.parse_args()
    
    # 确保使用绝对路径，避免相对路径问题
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Running GLUE on dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # 启动GPU监控（GLUE默认使用GPU）
    gpu_monitor = None
    gpu_id = 0
    if PYNVML_AVAILABLE and GPUMonitor:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_monitor = GPUMonitor(gpu_id=gpu_id, sampling_interval=1.0)
                gpu_monitor.start()
        except Exception as e:
            print(f"Warning: Failed to start GPU monitoring: {e}")
            gpu_monitor = None
    
    try:
        # 获取基因组文件
        genome_file = get_genome_file(args.dataset, args.genome_dir)
        
        # 加载原始数据
        rna, atac = load_raw_data(args.dataset, args.data_root)
        
        # 预处理
        rna = preprocess_rna(rna, genome_file)
        atac = preprocess_atac(atac)
        
        # 构建或加载先验图
        graph_file = output_dir / "prior.graphml.gz"
        if args.skip_graph and graph_file.exists():
            print(f"Loading existing graph from {graph_file}")
            graph = nx.read_graphml(graph_file)
        else:
            graph = build_prior_graph(rna, atac, output_dir)
        
        # 准备训练数据
        rna, atac = prepare_for_training(rna, atac)
        
        # 配置数据集
        rna, atac, graph = configure_datasets(rna, atac, graph)
        
        # 训练或加载模型
        glue = train_or_load_glue(rna, atac, graph, output_dir, train=args.train)
        
        # 编码数据并保存
        rna_embeddings, atac_embeddings = encode_and_save(glue, rna, atac, output_dir)
        
        # 生成 UMAP
        generate_umap(rna, atac, rna_embeddings, atac_embeddings, output_dir)
        
        print("=" * 60)
        print("GLUE completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # 停止GPU监控并保存统计数据
        if gpu_monitor and save_gpu_stats:
            try:
                stats = gpu_monitor.stop()
                # 保存GPU统计数据
                csv_path = Path(args.gpu_csv) if args.gpu_csv else None
                save_gpu_stats(
                    stats,
                    output_dir.parent,
                    method_name='glue',
                    gpu_id=gpu_id,
                    csv_path=csv_path
                )
            except Exception as e:
                print(f"Warning: Failed to save GPU stats: {e}")


if __name__ == '__main__':
    main()
