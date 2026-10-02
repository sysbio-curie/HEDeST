from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from torch import optim

from hedest.analysis.loaders import HedestRun
from hedest.analysis.loaders import INFO_NAME
from hedest.analysis.palette import Palette
from hedest.analysis.palette import palette_from_yaml
from hedest.analysis.pred_analyzer import PredAnalyzer
from hedest.analysis.stats import write_stats
from hedest.dataset import SpotEmbedDataset
from hedest.dataset_utils import custom_collate
from hedest.dataset_utils import split_data
from hedest.model.cell_classifier import CellClassifier
from hedest.ppsa import ADJUSTMENT_CLASSES
from hedest.ppsa import ADJUSTMENT_METHODS
from hedest.predict import predict_slide
from hedest.trainer import ModelTrainer
from hedest.utils import format_time
from hedest.utils import set_seed


def run_hedest(
    embed_dict: Dict[str, torch.Tensor],
    spot_prop_df: pd.DataFrame,
    spot_dict: Dict[str, List[str]],
    json_path: Optional[str] = None,
    adata: Optional[AnnData] = None,
    adata_name: Optional[str] = None,
    hidden_dims: List[int] = [1024, 512],
    norm: bool = True,
    dropout: float = 0.0,
    batch_size: int = 64,
    lr: float = 0.0001,
    divergence: str = "l2",
    alpha: float = 0.01,
    beta: float = 0.0,
    adjustment: str = "interpolated",
    gated: bool = False,
    epochs: int = 100,
    train_size: float = 0.8,
    val_size: float = 0.1,
    out_dir: str = "results",
    save_geojson: bool = False,
    color_dict_file: Optional[str] = None,
    rs: int = 42,
) -> None:
    """
    Runs HEDeST for cell classification.

    Args:
        embed_dict: Dictionary mapping cell IDs to cell embeddings.
        spot_prop_df: DataFrame containing cell type proportions for each spot.
        spot_dict: Dictionary mapping cell IDs to their spot.
        json_path: Path to the post-segmentation file.
        adata: AnnData object containing spatial transcriptomics data.
        adata_name: Name of the sample in the AnnData object.
        hidden_dims: List of hidden layer dimensions.
        norm: Whether to add a LayerNorm layer.
        dropout: Dropout rate.
        batch_size: Batch size for data loaders.
        lr: Learning rate for the optimizer.
        divergence: Type of divergence loss to use ("l1", "l2", "kl", "rot").
        alpha: Weighting factor for the loss function.
        beta: Weighting factor for the Bayesian adjustment.
        adjustment: PPSA method, "interpolated" (weighted mean of the <= 3 nearest spots for the
                    cells outside spots) or "nearest" (proportions of the closest spot).
        gated: If True, adjust only the cells inside spots. If False (default), adjust every cell,
               which needs adata, adata_name and json_path to locate the cells outside spots.
        epochs: Number of training epochs.
        train_size: Proportion of data used for training.
        val_size: Proportion of data used for validation.
        out_dir: Directory to save results.
        save_geojson: Whether to export a GeoJSON file for QuPath.
        color_dict_file: Path to a YAML color dict (special format).
        rs: Random seed for reproducibility.
    """

    if adjustment not in ADJUSTMENT_METHODS:
        raise ValueError(f"Invalid value for 'adjustment': {adjustment}. Must be one of {set(ADJUSTMENT_METHODS)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f"Created output directory: {out_dir}")

    missing_spots = set(spot_dict.keys()) - set(spot_prop_df.index)
    assert not missing_spots, (
        f"{len(missing_spots)} spot(s) in spot_dict have no proportions in spot_prop_df "
        f"(e.g. {sorted(missing_spots)[:5]}). Every spot with cells must have a proportion row."
    )

    train_spot_dict, train_proportions, val_spot_dict, val_proportions, test_spot_dict, test_proportions = split_data(
        spot_dict, spot_prop_df, train_size=train_size, val_size=val_size, rs=rs
    )

    # Create datasets
    set_seed(rs)
    logger.debug("Creating datasets...")
    train_dataset = SpotEmbedDataset(train_spot_dict, train_proportions, embed_dict)
    val_dataset = SpotEmbedDataset(val_spot_dict, val_proportions, embed_dict)
    test_dataset = SpotEmbedDataset(test_spot_dict, test_proportions, embed_dict)

    embed_size = train_dataset.embed_size

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
    )

    num_classes = spot_prop_df.shape[1]
    ct_list = list(spot_prop_df.columns)

    # Model initialization
    model = CellClassifier(
        num_classes=num_classes,
        embed_size=embed_size,
        hidden_dims=hidden_dims,
        norm=norm,
        dropout=dropout,
        device=device,
    )
    model = model.to(device)
    logger.info(f"-> {num_classes} classes detected.")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Model training
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        divergence=divergence,
        alpha=alpha,
        num_epochs=epochs,
        out_dir=out_dir,
        rs=rs,
    )

    logger.info("Starting training...")
    TRAIN_START = time.time()
    trainer.train()
    trainer.save_history()
    TRAIN_TIME = format_time(time.time() - TRAIN_START)
    logger.info("Training completed.")

    # Predict on the whole slide
    logger.info("Starting prediction on the whole slide...")
    model4pred_best = CellClassifier(
        num_classes=num_classes,
        embed_size=embed_size,
        hidden_dims=hidden_dims,
        norm=norm,
        dropout=dropout,
        device=device,
    )
    model4pred_best.load_state_dict(torch.load(trainer.best_model_path))
    cell_prob_best = predict_slide(model4pred_best, embed_dict, ct_list)

    # Prior Probability Shift adjustment
    p_c = spot_prop_df.loc[list(train_spot_dict.keys())].mean(axis=0)

    logger.info(f"Starting {adjustment} PPSA (gated={gated})...")
    ppsa = ADJUSTMENT_CLASSES[adjustment](
        cell_prob_best,
        spot_dict,
        spot_prop_df,
        p_c,
        adata=adata,
        adata_name=adata_name,
        json_path=json_path,
        gated=gated,
        beta=beta,
        device=device,
    )
    cell_prob_best_adjusted = ppsa.adjust()
    logger.info(
        f"-> {len(ppsa.adjustable_cells)}/{len(cell_prob_best)} cells adjusted "
        f"(method={ppsa.method}, gated={ppsa.gated})."
    )

    # Save everything the analysis package needs to re-open this run
    run = HedestRun(
        predictions_raw=cell_prob_best,
        predictions_adjusted=cell_prob_best_adjusted,
        spot_dict=spot_dict,
        proportions=spot_prop_df,
        params={
            "hidden_dims": hidden_dims,
            "norm": norm,
            "dropout": dropout,
            "embed_size": embed_size,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "lr": lr,
            "divergence": divergence,
            "alpha": alpha,
            "beta": beta,
            "adjustment": ppsa.method,  # the PPSA settings actually applied
            "gated": ppsa.gated,
            "epochs": epochs,
            "train_size": train_size,
            "val_size": val_size,
            "rs": rs,
            "train_time": TRAIN_TIME,
        },
        history={"train": trainer.history_train, "val": trainer.history_val},
        train_spot_dict=train_spot_dict,
        run_dir=Path(out_dir),
        seeds=[rs],
    )
    run.save(Path(out_dir) / INFO_NAME)

    logger.info("Computing the run statistics...")
    write_stats(Path(out_dir) / "stats.xlsx", run)

    # GeoJSON export for QuPath
    if save_geojson and json_path is None:
        logger.warning("save_geojson=True but json_path is None: no cell contours, so nothing is exported.")
    elif save_geojson:
        import yaml

        if color_dict_file is not None:
            with open(color_dict_file) as color_file:
                palette = palette_from_yaml(yaml.safe_load(color_file))
        else:
            palette = Palette(ct_list)
            auto_color_path = os.path.join(out_dir, "auto_color_dict.yaml")
            with open(auto_color_path, "w") as color_file:
                yaml.dump(palette.as_geojson_dict(), color_file)
            logger.info(f"Colour dictionary saved to {auto_color_path}")

        for adjusted, name in ((True, "hedest_predictions_adj.geojson"), (False, "hedest_predictions.geojson")):
            analyzer = PredAnalyzer(run, seg=json_path, palette=palette, adjusted=adjusted)
            analyzer.export_geojson(Path(out_dir) / name)

    logger.info("Secondary deconvolution process completed successfully.")
    logger.info(f"Training time: {TRAIN_TIME}")
