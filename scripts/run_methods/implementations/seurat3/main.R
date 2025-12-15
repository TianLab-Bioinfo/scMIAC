#!/usr/bin/env Rscript
# Seurat3 方法运行脚本
# 使用 Seurat v3/v4 进行 RNA-ATAC 多模态整合
#
# Usage:
#   Rscript main.R --dataset 10x --output-dir data/10x/output/methods/seurat3
#   Rscript main.R --dataset 10x --output-dir data/10x/output/methods/seurat3 --data-root data

suppressPackageStartupMessages({
  library(Seurat)
  library(Signac)
  library(dplyr)
  library(optparse)
})

set_random_seed <- function(seed = 24) {
  set.seed(seed)
}

option_list <- list(
  make_option(c("--dataset"), type = "character", default = NULL,
              help = "Dataset name", metavar = "character"),
  make_option(c("--output-dir"), type = "character", default = NULL,
              help = "Output directory", metavar = "character"),
  make_option(c("--data-root"), type = "character", default = "data",
              help = "Data root directory [default: %default]", metavar = "character"),
  make_option(c("--seed"), type = "integer", default = 24,
              help = "Random seed [default: %default]", metavar = "integer")
)

opt_parser <- OptionParser(option_list = option_list,
                          description = "\nRun Seurat3 method for multi-modal integration")
opt <- parse_args(opt_parser)

if (is.null(opt$dataset) || is.null(opt$`output-dir`)) {
  print_help(opt_parser)
  stop("Arguments --dataset and --output-dir are required.", call. = FALSE)
}

dataset_name <- opt$dataset
output_dir <- opt$`output-dir`
data_root <- opt$`data-root`
seed <- opt$seed

cat("============================================================\n")
cat(sprintf("Running Seurat3 on dataset: %s\n", dataset_name))
cat(sprintf("Output directory: %s\n", output_dir))
cat("============================================================\n")

tryCatch({
  set_random_seed(seed)
  
  cat("Loading data...\n")
  data_dir <- file.path(data_root, dataset_name, "input")
  rna_path <- file.path(data_dir, paste0("SeuratObj_RNA_", dataset_name, ".RDS"))
  atac_path <- file.path(data_dir, paste0("SeuratObj_ATAC_", dataset_name, ".RDS"))
  
  if (!file.exists(rna_path)) {
    stop(sprintf("RNA data not found: %s", rna_path))
  }
  if (!file.exists(atac_path)) {
    stop(sprintf("ATAC data not found: %s", atac_path))
  }
  
  # 读取 RDS 文件并立即更新对象以兼容 SeuratObject 5.0.0+
  SeuratObj_RNA <- readRDS(rna_path)
  SeuratObj_RNA <- UpdateSeuratObject(SeuratObj_RNA)
  SeuratObj_RNA$cell_type <- SeuratObj_RNA$true
  SeuratObj_RNA$modality <- "RNA"
  
  SeuratObj_ATAC <- readRDS(atac_path)
  SeuratObj_ATAC <- UpdateSeuratObject(SeuratObj_ATAC)
  SeuratObj_ATAC$cell_type <- SeuratObj_ATAC$true
  SeuratObj_ATAC$modality <- "ATAC"
  
  cat(sprintf("RNA: %d cells x %d features\n", ncol(SeuratObj_RNA), nrow(SeuratObj_RNA)))
  cat(sprintf("ATAC: %d cells x %d features\n", ncol(SeuratObj_ATAC), nrow(SeuratObj_ATAC)))
  
  # 预处理 RNA 数据
  cat("Preprocessing RNA data...\n")
  DefaultAssay(SeuratObj_RNA) <- "RNA"
  SeuratObj_RNA <- FindVariableFeatures(SeuratObj_RNA, nfeatures = 2000)
  
  # 预处理 ATAC 数据
  cat("Preprocessing ATAC data...\n")
  DefaultAssay(SeuratObj_ATAC) <- "ATAC"
  
  # 寻找转移锚点
  cat("Finding transfer anchors...\n")
  transfer.anchors <- FindTransferAnchors(
    reference = SeuratObj_RNA,
    query = SeuratObj_ATAC,
    features = VariableFeatures(SeuratObj_RNA),
    reference.assay = "RNA",
    query.assay = "ACTIVITY",
    reduction = "cca"
  )
  
  cat("Transfer anchors found\n")
  
  # 进行数据插补和联合嵌入
  cat("Performing data imputation...\n")
  genes.use <- VariableFeatures(SeuratObj_RNA)
  
  # 使用新版 API 获取数据
  refdata <- GetAssayData(SeuratObj_RNA, assay = "RNA", layer = "data")[genes.use, ]
  
  # 为 ATAC 细胞插补 RNA 表达
  imputation <- TransferData(
    anchorset = transfer.anchors,
    refdata = refdata,
    weight.reduction = "cca",
    dims = 1:30
  )
  
  SeuratObj_ATAC[["RNA"]] <- imputation
  
  # 合并两个模态
  cat("Merging modalities...\n")
  SeuratObj_ATAC$modality <- "ATAC"
  SeuratObj_RNA$modality <- "RNA"
  
  coembed <- merge(x = SeuratObj_RNA, y = SeuratObj_ATAC)
  
  # 联合降维
  cat("Running joint dimensionality reduction...\n")
  DefaultAssay(coembed) <- "RNA"
  coembed <- ScaleData(coembed, features = genes.use, do.scale = FALSE)
  coembed <- RunPCA(coembed, features = genes.use, verbose = FALSE)
  coembed <- RunUMAP(coembed, dims = 1:30, verbose = FALSE)
  
  # 保存结果
  cat("Saving results...\n")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # 提取并保存嵌入
  rna_embedding <- Embeddings(coembed, reduction = "pca")[coembed$modality == "RNA", ]
  atac_embedding <- Embeddings(coembed, reduction = "pca")[coembed$modality == "ATAC", ]
  
  rna_output <- file.path(output_dir, "rna_embeddings.csv")
  atac_output <- file.path(output_dir, "atac_embeddings.csv")
  
  write.csv(rna_embedding, rna_output)
  write.csv(atac_embedding, atac_output)
  
  cat(sprintf("RNA embeddings saved to %s\n", rna_output))
  cat(sprintf("ATAC embeddings saved to %s\n", atac_output))
  
  # 保存完整对象
  coembed_output <- file.path(output_dir, "coembed.RDS")
  saveRDS(coembed, coembed_output)
  cat(sprintf("Combined object saved to %s\n", coembed_output))
  
  # 生成并保存 UMAP 可视化
  cat("Generating UMAP visualization...\n")
  umap_png <- file.path(output_dir, "seurat3_latent_umap.png")
  
  # 检查是否有 cell_type 列
  color_by <- if ("cell_type" %in% colnames(coembed@meta.data)) {
    c("cell_type", "modality")
  } else {
    c("modality")
  }
  
  png(umap_png, width = 12, height = 6, units = "in", res = 150)
  print(DimPlot(coembed, reduction = "umap", group.by = color_by, label = TRUE))
  dev.off()
  
  cat(sprintf("UMAP plot saved to %s\n", umap_png))
  
  cat("============================================================\n")
  cat("Seurat3 completed successfully!\n")
  cat("============================================================\n")
  
}, error = function(e) {
  cat(sprintf("Error: %s\n", e$message), file = stderr())
  traceback()
  quit(status = 1)
})
