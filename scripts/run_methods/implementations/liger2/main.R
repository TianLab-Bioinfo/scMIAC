#!/usr/bin/env Rscript
# LIGER2 方法运行脚本
# 使用 rliger 进行 RNA-ATAC 多模态整合
#
# Usage:
#   Rscript main.R --dataset 10x --output-dir data/10x/output/methods/liger2
#   Rscript main.R --dataset 10x --output-dir data/10x/output/methods/liger2 --data-root data

suppressPackageStartupMessages({
  library(rliger)
  library(Seurat)
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
  make_option(c("--k"), type = "integer", default = 20,
              help = "Number of factors for LIGER [default: %default]", metavar = "integer"),
  make_option(c("--seed"), type = "integer", default = 24,
              help = "Random seed [default: %default]", metavar = "integer")
)

opt_parser <- OptionParser(option_list = option_list,
                          description = "\nRun LIGER2 method for multi-modal integration")
opt <- parse_args(opt_parser)

if (is.null(opt$dataset) || is.null(opt$`output-dir`)) {
  print_help(opt_parser)
  stop("Arguments --dataset and --output-dir are required.", call. = FALSE)
}

dataset_name <- opt$dataset
output_dir <- opt$`output-dir`
data_root <- opt$`data-root`
k_factors <- opt$k
seed <- opt$seed

cat("============================================================\n")
cat(sprintf("Running LIGER2 on dataset: %s\n", dataset_name))
cat(sprintf("Output directory: %s\n", output_dir))
cat(sprintf("Number of factors (k): %d\n", k_factors))
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
  
  # 读取 RDS 文件
  SeuratObj_RNA <- readRDS(rna_path)
  SeuratObj_RNA$cell_type <- SeuratObj_RNA$true
  SeuratObj_RNA$modality <- "RNA"
  
  SeuratObj_ATAC <- readRDS(atac_path)
  SeuratObj_ATAC$cell_type <- SeuratObj_ATAC$true
  SeuratObj_ATAC$modality <- "ATAC"
  
  cat(sprintf("RNA: %d cells x %d features\n", ncol(SeuratObj_RNA), nrow(SeuratObj_RNA)))
  cat(sprintf("ATAC: %d cells x %d features\n", ncol(SeuratObj_ATAC), nrow(SeuratObj_ATAC)))
    
  # 预处理
  cat("Preprocessing data...\n")
  DefaultAssay(SeuratObj_RNA) <- "RNA"
  DefaultAssay(SeuratObj_ATAC) <- "ATAC"
  
  # ATAC 数据作为 RNA assay
  SeuratObj_ATAC[["RNA"]] <- SeuratObj_ATAC[["ACTIVITY"]]
  DefaultAssay(SeuratObj_ATAC) <- "RNA"
  
  # 创建 LIGER 对象
  cat("Creating LIGER object...\n")
  seurat_list <- list(rna = SeuratObj_RNA, atac = SeuratObj_ATAC)
  liger_obj <- createLiger(seurat_list, modal = c("rna", "atac"))
  
  # LIGER 流程
  cat("Running LIGER integration...\n")
  liger_obj <- normalize(liger_obj)
  liger_obj <- selectGenes(liger_obj, useDatasets = "rna")
  liger_obj <- scaleNotCenter(liger_obj)
  liger_obj <- runIntegration(liger_obj, k = k_factors)
  liger_obj <- quantileNorm(liger_obj)
  
  # 提取归一化后的嵌入
  H <- liger_obj@H.norm
  
  cat("Merging results...\n")
  coembed <- merge(SeuratObj_RNA, SeuratObj_ATAC, add.cell.ids = c("RNA", "ATAC"))
  
  # 确保 H 的行名与 coembed 的细胞名一致
  rownames(H) <- colnames(coembed)
  
  # 将 LIGER 嵌入添加到 Seurat 对象
  coembed[["liger"]] <- CreateDimReducObject(
    embeddings = H,
    key = "liger_",
    assay = "RNA"
  )
  
  # 运行 UMAP
  cat("Running UMAP...\n")
  coembed <- RunUMAP(coembed, dims = 1:k_factors, reduction = "liger", verbose = FALSE)
  
  # 保存结果
  cat("Saving results...\n")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # 提取并保存嵌入
  rna_embedding <- Embeddings(coembed, reduction = "liger")[coembed$modality == "RNA", ]
  atac_embedding <- Embeddings(coembed, reduction = "liger")[coembed$modality == "ATAC", ]
  
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
  umap_png <- file.path(output_dir, "liger2_latent_umap.png")
  
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
  cat("LIGER2 completed successfully!\n")
  cat("============================================================\n")
  
}, error = function(e) {
  cat(sprintf("Error: %s\n", e$message), file = stderr())
  traceback()
  quit(status = 1)
})
