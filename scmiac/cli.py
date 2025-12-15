from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from .modeling.scmiac import find_anchors, model_inference, preprocess, train_model
from .preprocessing.preprocess import run_umap


# Recommended directory structure:
# data/
# ├── {dataset}/
# │   ├── input/                     # Original input data
# │   │   ├── adata_rna_{dataset}.h5ad
# │   │   └── adata_atac_{dataset}.h5ad
# │   └── output/                    # All experimental outputs
# │       ├── methods/{method}/      # Benchmark method results
# │       ├── ablation/{exp}/        # Ablation experiment results
# │       └── hyperparameter/{param}/{value}/  # Hyperparameter experiment results


def require_argument(value: Any, name: str) -> None:
    if value is None:
        raise SystemExit(f"Missing required argument: {name}. Provide it via CLI or YAML config.")


def parse_hidden_dims(value: str) -> list[int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("hidden_dims must contain at least one integer")
    try:
        return [int(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden_dims must be a comma-separated list of integers") from exc


def run_train(args: argparse.Namespace) -> int:
    require_argument(args.rna_h5ad, "--rna-h5ad")
    require_argument(args.atac_h5ad, "--atac-h5ad")
    require_argument(args.output_dir, "--output-dir")

    adata_rna = sc.read_h5ad(args.rna_h5ad)
    adata_atac = sc.read_h5ad(args.atac_h5ad)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup output paths
    anchor_path = output_dir / "anchors.csv"
    rna_vae_path = output_dir / "rna_vae.pth"
    atac_vae_path = output_dir / "atac_vae.pth"
    rna_embeddings_path = output_dir / "rna_embeddings.csv"
    atac_embeddings_path = output_dir / "atac_embeddings.csv"
    umap_path = output_dir / "scmiac_latent_umap.png"
    
    # Generate MNN anchors (always regenerate, no caching)
    print("Generating MNN anchors...")
    anchor_df = find_anchors(
        adata_rna,
        adata_atac,
        all_nfeatures=args.all_nfeatures,
        single_nfeatures=args.single_nfeatures,
        k_anchor=args.k_anchor,
        n_components=args.n_components,
        ct_filter=True,  # Always enable cell type filtering in CLI
        mode=args.mode,
        rna_celltype_key=args.rna_celltype_key,
        atac_celltype_key=args.atac_celltype_key,
    )
    anchor_df.to_csv(anchor_path, index=False)
    print(f"MNN anchors saved to {anchor_path} ({len(anchor_df)} pairs)")

    rna_vae, atac_vae, all_cells_loader, anchor_cells_loader = preprocess(
        adata_rna,
        adata_atac,
        anchor_df,
        rna_latent_key=args.rna_latent_key,
        atac_latent_key=args.atac_latent_key,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        balanced_sampler=args.balanced_sampler,
        device=args.device,
    )

    rna_vae, atac_vae = train_model(
        rna_vae,
        atac_vae,
        all_cells_loader,
        anchor_cells_loader,
        device=args.device,
        num_epoches=args.num_epochs,
        lambda_rna_kl=args.lambda_rna_kl,
        lambda_atac_kl=args.lambda_atac_kl,
        alpha_rna_rec=args.alpha_rna_rec,
        alpha_atac_rec=args.alpha_atac_rec,
        lambda_contra=args.lambda_contra,
        temperature=args.temperature,
        anchor_loss_type='contrastive',  # Always use contrastive learning in CLI
        lr=args.learning_rate,
        print_step=args.print_step,
        save_model=False,
    )

    torch.save(rna_vae.state_dict(), rna_vae_path)
    torch.save(atac_vae.state_dict(), atac_vae_path)
    print(f"Model weights saved to {rna_vae_path} and {atac_vae_path}")

    # Generate embeddings
    print("Generating embeddings...")
    rna_embeddings, atac_embeddings = model_inference(
        rna_vae,
        atac_vae,
        all_cells_loader,
        device=args.device,
    )
    pd.DataFrame(rna_embeddings, index=adata_rna.obs_names).to_csv(rna_embeddings_path)
    pd.DataFrame(atac_embeddings, index=adata_atac.obs_names).to_csv(atac_embeddings_path)
    print(f"Embeddings saved to {rna_embeddings_path} and {atac_embeddings_path}")

    print("Generating UMAP visualization...")
    rna_latent = rna_embeddings
    atac_latent = atac_embeddings

    def prepare_adata(source: ad.AnnData, embeddings: np.ndarray, modality: str) -> ad.AnnData:
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

    prepared_rna = prepare_adata(adata_rna, rna_latent, "RNA")
    prepared_atac = prepare_adata(adata_atac, atac_latent, "ATAC")

    adata_cm = ad.concat([prepared_rna, prepared_atac], join="outer", index_unique=None)
    adata_cm = run_umap(adata_cm, "scmiac_latent")

    fig = sc.pl.embedding(
        adata_cm,
        basis="scmiac_latent_umap",
        color=["cell_type", "modality"],
        legend_loc="on data",
        show=False,
        return_fig=True,
    )
    fig.savefig(umap_path, bbox_inches="tight", dpi=150)
    print(f"UMAP plot saved to {umap_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="scMIAC: Single-Cell Multi-modality Integration via cell type filtered Anchors using Contrastive learning",
        epilog="Example: scmiac train --rna-h5ad data/10x/input/adata_rna_10x.h5ad --atac-h5ad data/10x/input/adata_atac_10x.h5ad "
               "--output-dir data/10x/output/methods/scmiac/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser(
        "train",
        help="Train scMIAC models with automatic anchor generation",
        description="Train VAE models for RNA and ATAC integration.",
    )
    train_parser.add_argument("--rna-h5ad", default=None, help="Path to RNA AnnData file (e.g., data/10x/input/adata_rna_10x.h5ad)")
    train_parser.add_argument("--atac-h5ad", default=None, help="Path to ATAC AnnData file (e.g., data/10x/input/adata_atac_10x.h5ad)")
    train_parser.add_argument("--output-dir", default=None, help="Output directory for results (anchors, models, embeddings, UMAP)")
    train_parser.add_argument("--all-nfeatures", type=int, default=3000, help="Number of features for integration when generating anchors")
    train_parser.add_argument("--single-nfeatures", type=int, default=2000, help="Number of features per modality when generating anchors")
    train_parser.add_argument("--k-anchor", type=int, default=5, help="Number of neighbors when selecting anchors")
    train_parser.add_argument("--n-components", type=int, default=30, help="Dimensionality for CCA during anchor finding")
    train_parser.add_argument("--mode", choices=["v"], default=None, help="Set to 'v' to use positional anchors for vertical integration (paired data)")
    train_parser.add_argument("--rna-latent-key", default="X_pca", help="Key for RNA latent representation in AnnData")
    train_parser.add_argument("--atac-latent-key", default="lsi49", help="Key for ATAC latent representation in AnnData")
    train_parser.add_argument("--rna-celltype-key", default="cell_type", help="Key for RNA cell type annotation in AnnData.obs")
    train_parser.add_argument("--atac-celltype-key", default="pred", help="Key for ATAC cell type annotation in AnnData.obs")
    train_parser.add_argument("--batch-size", type=int, default=256, help="Batch size for DataLoaders")
    train_parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=[128, 64], help="Comma-separated hidden dimensions for VAE")
    train_parser.add_argument("--latent-dim", type=int, default=30, help="Latent dimension size for VAE")
    train_parser.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false", help="Disable balanced anchor sampling")
    train_parser.add_argument("--device", default="cuda:0", help="Torch device to use, e.g. cpu or cuda:0")
    train_parser.add_argument("--num-epochs", type=int, default=2000, help="Training epochs")
    train_parser.add_argument("--lambda-rna-kl", type=float, default=1.0, help="Weight for RNA KL divergence")
    train_parser.add_argument("--lambda-atac-kl", type=float, default=1.0, help="Weight for ATAC KL divergence")
    train_parser.add_argument("--alpha-rna-rec", type=float, default=20.0, help="Weight for RNA reconstruction loss")
    train_parser.add_argument("--alpha-atac-rec", type=float, default=20.0, help="Weight for ATAC reconstruction loss")
    train_parser.add_argument("--lambda-contra", type=float, default=300.0, help="Weight for anchor alignment loss")
    train_parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for contrastive loss")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--print-step", type=int, default=10, help="Logging interval in epochs")
    train_parser.set_defaults(func=run_train, balanced_sampler=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]

    # Show help if no arguments provided
    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
