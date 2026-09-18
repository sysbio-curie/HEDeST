from __future__ import annotations

import json
import os
import time
from typing import List
from typing import Optional

import torch
import typer
from loguru import logger

from hedest.analysis.postseg import map_cells_to_spots
from hedest.dataset_utils import pp_prop
from hedest.ppsa import ADJUSTMENT_METHODS
from hedest.run_model import run_hedest
from hedest.utils import format_time
from hedest.utils import load_spatial_adata
from hedest.utils import update_spot_diameter

app = typer.Typer()


def normalize_none(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"none", ""}:
        return None
    return x


def parse_hidden_dims(hidden_dims: str) -> List[int]:
    """
    Parses the hidden_dims string into a list of integers.

    Args:
        hidden_dims: Comma-separated string of integers.

    Returns:
        List of integers representing hidden dimensions.
    """

    try:
        return [int(dim) for dim in hidden_dims.split(",")]
    except ValueError:
        typer.echo("'hidden_dims' should be a str chain of integers separated by comas (e.g. '512,258').", err=True)
        raise typer.Abort()


@app.command()
def main(
    image_path: str = typer.Argument(..., help="Path to the high-quality WSI directory or image dict."),
    spot_prop_file: str = typer.Argument(..., help="Path to the proportions file."),
    json_path: Optional[str] = typer.Option(None, help="Path to the post-segmentation file."),
    path_st_adata: Optional[str] = typer.Option(None, help="Path to the ST anndata object."),
    adata_name: Optional[str] = typer.Option(None, help="Name of the sample."),
    spot_dict_file: Optional[str] = typer.Option(None, help="Path to the spot-to-cell json file."),
    mpp: Optional[float] = typer.Option(None, help="Microns per pixel of the WSI."),
    model_name: str = typer.Option("default", help="Type of model. Can be 'default', 'convnet', or 'resnet18'."),
    hidden_dims: str = typer.Option("512,256", help="Hidden dimensions for the model (comma-separated)."),
    norm: bool = typer.Option(False, help="Whether to add a LayerNorm layer."),
    dropout: float = typer.Option(0.0, help="Dropout rate."),
    batch_size: int = typer.Option(64, help="Batch size for model training."),
    lr: float = typer.Option(0.0001, help="Learning rate."),
    divergence: str = typer.Option("l2", help="Metric to use for divergence computation. Can be 'l1', 'l2' or 'kl'."),
    alpha: float = typer.Option(0.0, help="Alpha parameter for loss function."),
    beta: float = typer.Option(0.0, help="Beta parameter for bayesian adjustment."),
    adjustment: str = typer.Option(
        "interpolated",
        help=(
            "PPSA method for the cells outside spots: 'interpolated' (weighted mean of the <=3 "
            "nearest spots) or 'nearest' (proportions of the closest spot)."
        ),
    ),
    gated: bool = typer.Option(False, help="Adjust only the cells inside spots (needs no cell coordinates)."),
    epochs: int = typer.Option(60, help="Number of training epochs."),
    train_size: float = typer.Option(0.7, help="Training set size as a fraction."),
    val_size: float = typer.Option(0.15, help="Validation set size as a fraction."),
    out_dir: str = typer.Option("results", help="Output directory."),
    save_geojson: bool = typer.Option(False, help="Whether to export a GeoJSON file for QuPath."),
    color_dict_file: Optional[str] = typer.Option(None, help="Path to a YAML color dict (special format)."),
    rs: int = typer.Option(42, help="Random seed"),
):

    json_path = normalize_none(json_path)
    path_st_adata = normalize_none(path_st_adata)
    adata_name = normalize_none(adata_name)
    spot_dict_file = normalize_none(spot_dict_file)
    mpp = normalize_none(mpp)
    color_dict_file = normalize_none(color_dict_file)
    hidden_dims = parse_hidden_dims(hidden_dims)

    # Validate inputs
    valid_divergence = {"l1", "l2", "kl"}
    valid_model_name = {"default", "convnet", "resnet18"}
    valid_adjustment = set(ADJUSTMENT_METHODS)

    if divergence not in valid_divergence:
        raise ValueError(f"Invalid value for 'divergence': {divergence}. Must be one of {valid_divergence}.")
    if model_name not in valid_model_name:
        raise ValueError(f"Invalid value for 'model_name': {model_name}. Must be one of {valid_model_name}.")
    if adjustment not in valid_adjustment:
        raise ValueError(f"Invalid value for 'adjustment': {adjustment}. Must be one of {valid_adjustment}.")

    MAIN_START = time.time()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f"-> Created output directory: {out_dir}")

    # Image data loading
    if image_path.endswith(".pt"):
        logger.info(f"-> Loading image dictionary from {image_path}")
        image_dict = torch.load(image_path)

    else:
        raise ValueError(
            f"Invalid image_path: {image_path}. "
            "Expected a .pt file containing an image dictionary. "
            "If you tried to pass a WSI directly, please segment it first "
            "with run_hovernet.sh."
        )

    example_img = image_dict[list(image_dict.keys())[0]]
    try:
        size = (example_img.shape[1], example_img.shape[1])
    except Exception:
        size = example_img.shape[0]

    # Load spot information
    logger.info(f"Loading proportions from {spot_prop_file}...")
    spot_prop_df = pp_prop(spot_prop_file)

    adata = None
    spot_dict = None
    if path_st_adata is not None:
        logger.info(f"Loading spatial transcriptomics data from {path_st_adata}...")
        adata = load_spatial_adata(path_st_adata)
    else:
        logger.info("No spatial transcriptomics data provided. There will be no spatial PPS adjustment...")

    if adata is not None and adata_name is None:
        try:
            adata_name = list(adata.uns["spatial"].keys())[0]
            logger.info(f"-> Using default adata_name: {adata_name}")
        except Exception:
            raise ValueError("adata_name was not provided and could not be inferred from adata.uns['spatial'].")

    if adata is not None and mpp is not None:
        logger.info(f"Applying spot diameter update using mpp={mpp}...")
        adata = update_spot_diameter(adata, adata_name, mpp)

    if spot_dict_file is not None and os.path.splitext(spot_dict_file)[1] == ".json":
        logger.info(f"Loading spot-to-cell dictionary from {spot_dict_file}...")
        with open(spot_dict_file) as json_file:
            spot_dict = json.load(json_file)
    elif adata is not None and adata_name is not None and json_path is not None:
        logger.info("Mapping cells to the spot in which they are located...")
        spot_dict = map_cells_to_spots(adata, adata_name, json_path)
    else:
        raise ValueError(
            "To map cells to spots, please provide a valid spot_dict_file or provide adata, adata_name and json_path."
        )

    # Recap variables
    logger.info("=" * 50)
    logger.info("RUNNING SECONDARY DECONVOLUTION")
    logger.info("Parameters:")
    logger.info(f"Image size: {size}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Hidden dims: {hidden_dims}")
    logger.info(f"Normalization: {norm}")
    logger.info(f"Dropout: {dropout}")
    logger.info(f"Batch size (#spots): {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Divergence: {divergence}")
    logger.info(f"Alpha: {alpha}")
    logger.info(f"Beta: {beta}")
    logger.info(f"Adjustment: {adjustment} (gated: {gated})")
    logger.info(f"Number of epochs: {epochs}")
    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {val_size}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Random state: {rs}")
    logger.info("=" * 50)

    # Run HEDeST
    run_hedest(
        image_dict=image_dict,
        spot_prop_df=spot_prop_df,
        spot_dict=spot_dict,
        json_path=json_path,
        adata=adata,
        adata_name=adata_name,
        model_name=model_name,
        hidden_dims=hidden_dims,
        norm=norm,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        divergence=divergence,
        alpha=alpha,
        beta=beta,
        adjustment=adjustment,
        gated=gated,
        epochs=epochs,
        train_size=train_size,
        val_size=val_size,
        out_dir=out_dir,
        save_geojson=save_geojson,
        color_dict_file=color_dict_file,
        rs=rs,
    )
    TOTAL_TIME = format_time(time.time() - MAIN_START)
    logger.info(f"Total time: {TOTAL_TIME}\n")


if __name__ == "__main__":
    app()
