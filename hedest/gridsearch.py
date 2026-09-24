from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from typing import List

from loguru import logger


def run_experiment(
    features_path: str,
    spot_prop_df: str,
    json_path: str,
    path_st_adata: str,
    adata_name: str,
    spot_dict_file: str,
    hidden_dim: str,
    norm: bool,
    dropout: float,
    batch_size: int,
    alpha: float,
    beta: float,
    lr: float,
    divergence: str,
    out_dir: str,
    seed: int,
    adjustment: str = "interpolated",
    gated: bool = False,
) -> None:
    """
    Runs one experiment (model) with the specified parameters.

    Args:
        features_path: Path to the cell feature dictionary (.pt).
        spot_prop_df: Path to the spot proportions DataFrame.
        json_path: Path to the JSON file containing segmentation.
        path_st_adata: Path to the spatial transcriptomics AnnData file.
        adata_name: Name of the AnnData object.
        spot_dict_file: Path to the spot dictionary file.
        hidden_dim: Hidden dimensions for the model.
        norm: Whether to add a LayerNorm layer.
        dropout: Dropout rate.
        batch_size: Batch size for training.
        alpha: Regularization parameter for the model.
        beta: Regularization parameter for bayesian adjustment.
        lr: Learning rate for training.
        divergence: Divergence metric to use.
        out_dir: Output directory path.
        seed: Random seed for reproducibility.
        adjustment: PPSA method, "interpolated" or "nearest".
        gated: Whether PPSA adjusts only the cells inside spots.
    """

    config_out_dir = os.path.join(
        out_dir,
        (
            "model_default_"
            f"hidden_dim_{hidden_dim.replace(',', '-')}_"
            f"norm_{norm}_"
            f"dropout_{dropout}_"
            f"alpha_{alpha}_"
            f"lr_{lr}_"
            f"divergence_{divergence}_"
            f"beta_{beta}_"
            f"seed_{seed}"
        ),
    )
    os.makedirs(config_out_dir, exist_ok=True)

    args = [
        "python3",
        "-u",
        "hedest/main.py",
        features_path,
        spot_prop_df,
        "--json-path",
        json_path,
        "--path-st-adata",
        path_st_adata,
        "--adata-name",
        adata_name,
        "--spot-dict-file",
        spot_dict_file,
        "--hidden-dims",
        hidden_dim,
        "--dropout",
        str(dropout),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--divergence",
        divergence,
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--adjustment",
        adjustment,
        "--epochs",
        "80",
        "--out-dir",
        config_out_dir,
        "--rs",
        str(seed),
    ]

    if norm:
        args.append("--norm")
    if gated:
        args.append("--gated")

    subprocess.run(args, check=True)


def main_simulation(
    features_path: str,
    spot_prop_df: str,
    json_path: str,
    path_st_adata: str,
    adata_name: str,
    spot_dict_file: str,
    hidden_dims: List[str],
    norms: List[bool],
    dropouts: List[float],
    alphas: List[float],
    betas: List[float],
    learning_rates: List[float],
    divergences: List[str],
    seeds: List[int],
    batch_size: int,
    out_dir: str,
    adjustment: str = "interpolated",
    gated: bool = False,
) -> None:
    """
    Performs the main simulation pipeline for a given divergence metric.

    Args:
        features_path: Path to the cell feature dictionary (.pt).
        spot_prop_df: Path to the spot proportions DataFrame.
        json_path: Path to the JSON file containing segmentation.
        path_st_adata: Path to the spatial transcriptomics AnnData file.
        adata_name: Name of the AnnData object.
        spot_dict_file: Path to the spot dictionary file.
        hidden_dims: List of hidden dimensions.
        norms: List of normalization options.
        dropouts: List of dropout rates.
        alphas: List of alpha values.
        betas: List of beta values.
        learning_rates: List of learning rates.
        divergences: List of divergence metrics.
        seeds: List of random seed values.
        batch_size: Batch size for training.
        out_dir: Output directory path.
        adjustment: PPSA method, "interpolated" or "nearest".
        gated: Whether PPSA adjusts only the cells inside spots.
    """

    logger.info(f"Cell features path: {features_path}")
    logger.info(f"Spot proportions DataFrame path: {spot_prop_df}")
    logger.info(f"JSON path: {json_path}")
    logger.info(f"Path to spatial transcriptomics AnnData: {path_st_adata}")
    logger.info(f"AnnData name: {adata_name}")
    logger.info(f"Spot dictionary file path: {spot_dict_file}")
    logger.info(f"Hidden dimensions: {hidden_dims}")
    logger.info(f"Normalization options: {norms}")
    logger.info(f"Dropout rates: {dropouts}")
    logger.info(f"Alpha values: {alphas}")
    logger.info(f"Beta values: {betas}")
    logger.info(f"Learning rates: {learning_rates}")
    logger.info(f"Divergence metrics: {divergences}")
    logger.info(f"Random seeds: {seeds}")
    logger.info(f"Adjustment: {adjustment} (gated: {gated})")
    logger.info(f"Output directory: {out_dir}\n")

    combinations = list(itertools.product(hidden_dims, norms, dropouts, alphas, learning_rates, divergences, betas))

    for hidden_dim, norm, dropout, alpha, lr, divergence, beta in combinations:
        for seed in seeds:
            run_experiment(
                features_path,
                spot_prop_df,
                json_path,
                path_st_adata,
                adata_name,
                spot_dict_file,
                hidden_dim,
                norm,
                dropout,
                batch_size,
                alpha,
                beta,
                lr,
                divergence,
                out_dir,
                seed,
                adjustment=adjustment,
                gated=gated,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with specified parameters")

    # String arguments
    parser.add_argument("features_path", type=str, help="Path to the cell feature dictionary (.pt)")
    parser.add_argument("spot_prop_df", type=str, help="Path to the spot proportions DataFrame")
    parser.add_argument("json_path", type=str, help="Path to the JSON file containing segmentation")
    parser.add_argument("path_st_adata", type=str, help="Path to the spatial transcriptomics AnnData file")
    parser.add_argument("adata_name", type=str, help="Name of the AnnData object")
    parser.add_argument("spot_dict_file", type=str, help="Path to the spot dictionary file")

    # List arguments
    parser.add_argument("--hidden_dims", nargs="+", type=str, required=True, help="List of hidden dimensions")
    parser.add_argument(
        "--norm_options",
        nargs="+",
        type=int,
        required=True,
        choices=[0, 1],
        help="List of norm options (0 for False, 1 for True)",
    )
    parser.add_argument("--dropouts", nargs="+", type=float, required=True, help="List of dropout rates")
    parser.add_argument("--alphas", nargs="+", type=float, required=True, help="List of alpha values")
    parser.add_argument("--betas", nargs="+", type=float, required=True, help="List of beta values")
    parser.add_argument("--learning_rates", nargs="+", type=float, required=True, help="List of learning rates")
    parser.add_argument("--divergences", nargs="+", type=str, required=True, help="List of divergence metrics")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="List of random seed values")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument(
        "--adjustment",
        type=str,
        default="interpolated",
        choices=["interpolated", "nearest"],
        help="PPSA method used for the cells outside spots",
    )
    parser.add_argument("--gated", action="store_true", help="Adjust only the cells inside spots")

    # Output directory
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory path")

    args = parser.parse_args()
    norms = [bool(n) for n in args.norm_options]

    main_simulation(
        args.features_path,
        args.spot_prop_df,
        args.json_path,
        args.path_st_adata,
        args.adata_name,
        args.spot_dict_file,
        args.hidden_dims,
        norms,
        args.dropouts,
        args.alphas,
        args.betas,
        args.learning_rates,
        args.divergences,
        args.seeds,
        args.batch_size,
        args.out_dir,
        args.adjustment,
        args.gated,
    )
