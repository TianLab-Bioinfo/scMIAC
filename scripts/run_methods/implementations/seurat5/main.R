#!/usr/bin/env Rscript
# Seurat v5 Bridge Integration 方法运行脚本
# 使用 Seurat v5 的 bridge integration 进行 RNA-ATAC 多模态整合
#
# Bridge integration 原理：
#   使用部分配对的多组学数据（bridge/multiome data）作为"桥梁"，
#   将未配对的 RNA-only 和 ATAC-only 数据投影到共享的潜在空间。
#
# 参考：https://satijalab.org/seurat/articles/seurat5_integration_bridge
#
# Usage:
#   Rscript main.R --dataset 10x --output-dir data/10x/output/methods/seurat5 --paired-cells-file data/10x/input/paired_cells/paired_0.2_cells.txt
#   Rscript main.R --dataset share --output-dir data/share/output/methods/seurat5 --paired-cells-file data/share/input/paired_cells/paired_0.3_cells.txt

# 增加 future 包的全局对象大小限制，以支持大数据集
# 设置为 Inf 表示无限制
options(future.globals.maxSize = Inf)

suppressPackageStartupMessages({
  library(Seurat)
  library(Signac)
  library(dplyr)
  library(optparse)
})

set_random_seed <- function(seed = 42) {
  set.seed(seed)
}

# 创建部分配对场景
# 从文件读取配对细胞列表，将数据拆分为：
#   - bridge: 文件中指定的配对细胞（multiome）
#   - rna_only: 剩余细胞的 RNA 单模态细胞
#   - atac_only: 剩余细胞的 ATAC 单模态细胞
create_partial_pairing <- function(rna_obj, atac_obj, paired_cells_file) {
  cat("\nCreating partial pairing scenario...\n")
  
  # 确保两个对象有相同的细胞
  common_cells <- intersect(colnames(rna_obj), colnames(atac_obj))
  n_cells <- length(common_cells)
  
  if (n_cells == 0) {
    stop("No common cells found between RNA and ATAC data")
  }
  
  cat(sprintf("  Total cells: %d\n", n_cells))
  
  # 从文件读取配对细胞列表
  cat(sprintf("  Reading paired cells from: %s\n", paired_cells_file))
  if (!file.exists(paired_cells_file)) {
    stop(sprintf("Paired cells file not found: %s", paired_cells_file))
  }
  
  paired_cells <- readLines(paired_cells_file)
  paired_cells <- paired_cells[nzchar(paired_cells)]  # 移除空行
  
  # 验证配对细胞是否在数据中
  paired_cells_set <- unique(paired_cells)
  common_cells_set <- unique(common_cells)
  
  invalid_cells <- setdiff(paired_cells_set, common_cells_set)
  if (length(invalid_cells) > 0) {
    stop(sprintf("Found %d cells in paired_cells_file that are not in the data. First few: %s",
                 length(invalid_cells), paste(head(invalid_cells, 5), collapse=", ")))
  }
  
  # 过滤出在数据中的配对细胞（保持文件中的顺序）
  paired_cells <- paired_cells[paired_cells %in% common_cells_set]
  
  # 找出非配对细胞
  unpaired_cells <- setdiff(common_cells, paired_cells)
  
  n_paired <- length(paired_cells)
  n_unpaired <- length(unpaired_cells)
  actual_ratio <- n_paired / n_cells
  
  cat(sprintf("  Paired cells from file: %d (%.1f%%)\n", n_paired, actual_ratio * 100))
  cat(sprintf("  Unpaired: %d (%.1f%%)\n", n_unpaired, (1 - actual_ratio) * 100))
  cat(sprintf("    → Will be split into %d RNA-only + %d ATAC-only\n", n_unpaired, n_unpaired))
  
  # 创建 bridge 对象（配对细胞）
  rna_paired <- rna_obj[, paired_cells]
  atac_paired <- atac_obj[, paired_cells]
  
  # 创建 RNA-only 对象（添加后缀以区分）
  rna_only <- rna_obj[, unpaired_cells]
  colnames(rna_only) <- paste0(colnames(rna_only), "-RNA")
  
  # 创建 ATAC-only 对象（添加后缀以区分）
  atac_only <- atac_obj[, unpaired_cells]
  colnames(atac_only) <- paste0(colnames(atac_only), "-ATAC")
  
  cat("  Data splitting completed\n")
  
  return(list(
    rna_paired = rna_paired,
    atac_paired = atac_paired,
    rna_only = rna_only,
    atac_only = atac_only
  ))
}

# 命令行参数解析
option_list <- list(
  make_option(c("--dataset"), type = "character", default = NULL,
              help = "Dataset name", metavar = "character"),
  make_option(c("--output-dir"), type = "character", default = NULL,
              help = "Output directory", metavar = "character"),
  make_option(c("--data-root"), type = "character", default = "data",
              help = "Data root directory [default: %default]", metavar = "character"),
  make_option(c("--paired-cells-file"), type = "character", default = NULL,
              help = "Path to file containing paired cell barcodes (one per line). Required.", metavar = "character"),
  make_option(c("--seed"), type = "integer", default = 42,
              help = "Random seed [default: %default]", metavar = "integer")
)

opt_parser <- OptionParser(option_list = option_list,
                          description = "\nRun Seurat v5 bridge integration for multi-modal integration")
opt <- parse_args(opt_parser)

if (is.null(opt$dataset) || is.null(opt$`output-dir`) || is.null(opt$`paired-cells-file`)) {
  print_help(opt_parser)
  stop("Arguments --dataset, --output-dir, and --paired-cells-file are required.", call. = FALSE)
}

dataset_name <- opt$dataset
output_dir <- opt$`output-dir`
data_root <- opt$`data-root`
paired_cells_file <- opt$`paired-cells-file`
seed <- opt$seed

cat("============================================================\n")
cat(sprintf("Running Seurat v5 Bridge Integration on dataset: %s\n", dataset_name))
cat(sprintf("Output directory: %s\n", output_dir))
cat(sprintf("Paired cells file: %s\n", paired_cells_file))
cat(sprintf("Seurat version: %s\n", packageVersion("Seurat")))
cat("============================================================\n")

tryCatch({
  set_random_seed(seed)
  
  # 加载数据
  cat("\nLoading data...\n")
  data_dir <- file.path(data_root, dataset_name, "input")
  rna_path <- file.path(data_dir, paste0("SeuratObj_RNA_", dataset_name, ".RDS"))
  atac_path <- file.path(data_dir, paste0("SeuratObj_ATAC_", dataset_name, ".RDS"))
  
  if (!file.exists(rna_path)) {
    stop(sprintf("RNA data not found: %s", rna_path))
  }
  if (!file.exists(atac_path)) {
    stop(sprintf("ATAC data not found: %s", atac_path))
  }
  
  # 读取并更新对象
  SeuratObj_RNA <- readRDS(rna_path)
  SeuratObj_RNA <- UpdateSeuratObject(SeuratObj_RNA)
  SeuratObj_RNA$cell_type <- SeuratObj_RNA$true
  SeuratObj_RNA$modality <- "RNA"
  
  SeuratObj_ATAC <- readRDS(atac_path)
  SeuratObj_ATAC <- UpdateSeuratObject(SeuratObj_ATAC)
  SeuratObj_ATAC$cell_type <- SeuratObj_ATAC$true
  SeuratObj_ATAC$modality <- "ATAC"

  cat(sprintf("Original RNA: %d cells x %d features\n", ncol(SeuratObj_RNA), nrow(SeuratObj_RNA)))
  cat(sprintf("Original ATAC: %d cells x %d features\n", ncol(SeuratObj_ATAC), nrow(SeuratObj_ATAC)))
  
  # 创建部分配对场景
  split_data <- create_partial_pairing(SeuratObj_RNA, SeuratObj_ATAC, paired_cells_file)
  
  # ========== 步骤1: 准备 Bridge 数据 (配对的多组学数据) ==========
  cat("\n=== Step 1: Preparing Bridge Data (Paired Multiome) ===\n")
  
  # 合并 RNA 和 ATAC 到一个 multiome 对象
  cat("Creating bridge multiome object...\n")
  bridge <- split_data$rna_paired
  
  # 添加 ATAC assay 到 bridge 对象
  bridge[["ATAC"]] <- CreateAssayObject(counts = GetAssayData(split_data$atac_paired, layer = "counts"))
  
  cat(sprintf("Bridge cells: %d\n", ncol(bridge)))
  cat(sprintf("Bridge assays: %s\n", paste(Assays(bridge), collapse = ", ")))
  
  # 预处理 bridge 的 RNA 数据
  cat("Preprocessing bridge RNA...\n")
  DefaultAssay(bridge) <- "RNA"
  bridge <- NormalizeData(bridge)
  bridge <- FindVariableFeatures(bridge, nfeatures = 2000)
  bridge <- ScaleData(bridge)
  bridge <- RunPCA(bridge, npcs = 50, verbose = FALSE)
  
  # 预处理 bridge 的 ATAC 数据
  cat("Preprocessing bridge ATAC...\n")
  DefaultAssay(bridge) <- "ATAC"
  bridge <- RunTFIDF(bridge)
  bridge <- FindTopFeatures(bridge, min.cutoff = 'q0')
  bridge <- RunSVD(bridge)
  
  # ========== 步骤2: 准备 Reference (未配对的 RNA 数据) ==========
  cat("\n=== Step 2: Preparing Reference (Unpaired RNA) ===\n")
  
  rna_reference <- split_data$rna_only
  DefaultAssay(rna_reference) <- "RNA"
  rna_reference <- NormalizeData(rna_reference)
  rna_reference <- FindVariableFeatures(rna_reference, nfeatures = 2000)
  rna_reference <- ScaleData(rna_reference)
  rna_reference <- RunPCA(rna_reference, npcs = 50, verbose = FALSE)
  cat(sprintf("RNA reference cells: %d\n", ncol(rna_reference)))
  
  # ========== 步骤3: 准备 Extended Reference (Reference + Bridge) ==========
  cat("\n=== Step 3: Preparing Extended Reference ===\n")
  
  # 使用 PrepareBridgeReference 创建扩展的 reference
  # 这会将 RNA reference 和 bridge 结合，使得可以映射 ATAC query
  cat("Creating extended reference with bridge...\n")
  rna_ext <- PrepareBridgeReference(
    reference = rna_reference,
    bridge = bridge,
    reference.reduction = "pca",
    reference.dims = 1:50,
    normalization.method = "LogNormalize"
  )
  
  cat("Extended reference prepared\n")
  
  # 检查 rna_ext 有哪些 reductions
  cat("Available reductions in rna_ext:", paste(names(rna_ext@reductions), collapse = ", "), "\n")
  
  # PrepareBridgeReference 可能创建了 "Bridge.reduc" 或重命名了 pca
  # 我们使用 Bridge.reduc 或第一个可用的 reduction
  if ("Bridge.reduc" %in% names(rna_ext@reductions)) {
    reduction_name <- "Bridge.reduc"
    cat("Using Bridge.reduc for UMAP\n")
  } else if ("pca.1" %in% names(rna_ext@reductions)) {
    reduction_name <- "pca.1"
    cat("Using pca.1 for UMAP\n")
  } else {
    reduction_name <- names(rna_ext@reductions)[1]
    cat("Using", reduction_name, "for UMAP\n")
  }
  
  # ========== 步骤4: 准备 Query (未配对的 ATAC 数据) ==========
  cat("\n=== Step 4: Preparing Query (Unpaired ATAC) ===\n")
  
  atac_query <- split_data$atac_only
  DefaultAssay(atac_query) <- "ATAC"
  atac_query <- RunTFIDF(atac_query)
  atac_query <- FindTopFeatures(atac_query, min.cutoff = 'q0')
  atac_query <- RunSVD(atac_query)
  cat(sprintf("ATAC query cells: %d\n", ncol(atac_query)))
  
  # ========== 步骤5: Bridge Integration - 整合 ATAC Query ==========
  cat("\n=== Step 5: Integrating ATAC Query via Bridge Integration ===\n")
  
  # 找到 ATAC query 到 extended reference 的整合锚点
  cat("Finding bridge integration anchors...\n")
  bridge_anchors <- FindBridgeIntegrationAnchors(
    extended.reference = rna_ext,
    query = atac_query,
    reduction = "lsiproject",
    dims = 2:50,
    verbose = FALSE
  )
  
  # 整合 embeddings - 将 ATAC query 和 extended reference 整合到统一空间
  # 这一步会自动处理所有细胞（RNA reference, bridge RNA/ATAC, ATAC query）
  cat("Integrating embeddings into unified space...\n")
  obj_integrated <- IntegrateEmbeddings(
    anchorset = bridge_anchors,
    verbose = FALSE
  )
  
  cat("Integration completed\n")
  cat("Available reductions in integrated object:", paste(names(obj_integrated@reductions), collapse = ", "), "\n")
  
  # ========== 步骤6: 整理数据并保存 ==========
  cat("\n=== Step 6: Saving Results ===\n")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # 从 obj_integrated 提取所有细胞的 integrated_dr embeddings
  cat("Extracting integrated embeddings from integrated_dr reduction...\n")
  integrated_emb_all <- Embeddings(obj_integrated, reduction = "integrated_dr")
  cat("All integrated embeddings shape:", nrow(integrated_emb_all), "x", ncol(integrated_emb_all), "\n")
  
  # 使用细胞名索引提取 embeddings
  # 注意：obj_integrated 只包含 rna_reference 和 atac_query，不包含 bridge 细胞
  rna_cells <- colnames(rna_reference)
  atac_cells <- colnames(atac_query)
  rna_embedding <- integrated_emb_all[rna_cells, ]
  atac_embedding <- integrated_emb_all[atac_cells, ]
  
  cat(sprintf("Extracted RNA embeddings: %d cells (unpaired RNA)\n", nrow(rna_embedding)))
  cat(sprintf("Extracted ATAC embeddings: %d cells (unpaired ATAC)\n", nrow(atac_embedding)))

  # 保存完整的对象
  rna_ext_output <- file.path(output_dir, "rna_ext.RDS")
  bridge_output <- file.path(output_dir, "bridge.RDS")
  obj_integrated_output <- file.path(output_dir, "obj_integrated.RDS")

  saveRDS(rna_ext, rna_ext_output)
  saveRDS(bridge, bridge_output)
  saveRDS(obj_integrated, obj_integrated_output)
  
  # 保存 embeddings
  rna_output <- file.path(output_dir, "rna_embeddings.csv")
  atac_output <- file.path(output_dir, "atac_embeddings.csv")
  
  write.csv(rna_embedding, rna_output)
  write.csv(atac_embedding, atac_output)

  # 生成并保存 UMAP 可视化
  cat("\nGenerating UMAP visualization...\n")
  obj_integrated <- RunUMAP(obj_integrated, reduction = "integrated_dr", dims = 1:50, return.model = TRUE, verbose = FALSE)

  umap_png <- file.path(output_dir, "seurat5_latent_umap.png")
  png(umap_png, width = 12, height = 6, units = "in", res = 150)
  print(DimPlot(obj_integrated,group.by = c("cell_type","modality"),label = T))
  dev.off()
  
  cat(sprintf("UMAP plot saved to %s\n", umap_png))
  
  cat("\n============================================================\n")
  cat("Seurat v5 Bridge Integration completed successfully!\n")
  cat("============================================================\n")
  
}, error = function(e) {
  cat(sprintf("Error: %s\n", e$message), file = stderr())
  traceback()
  quit(status = 1)
})
