from __future__ import annotations

import json
import os
import pickle
import time
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from torch import optim

from hedest.analysis.pred_analyzer import PredAnalyzer
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
    hidden_dims: List[int] = [512, 256],
    norm: bool = False,
    dropout: float = 0.0,
    batch_size: int = 64,
    lr: float = 0.0001,
    divergence: str = "l2",
    alpha: float = 0.0,
    beta: float = 0.0,
    adjustment: str = "interpolated",
    gated: bool = False,
    epochs: int = 60,
    train_size: float = 0.5,
    val_size: float = 0.25,
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

    # Save model infos
    model_info = {
        "model_name": "default",  # kept for the files written by earlier versions
        "hidden_dims": hidden_dims,
        "norm": norm,
        "dropout": dropout,
        "spot_dict": spot_dict,
        "train_spot_dict": train_spot_dict,
        "proportions": spot_prop_df,
        "adjustment": ppsa.method,  # PPSA settings actually applied
        "gated": ppsa.gated,
        "history": {"train": trainer.history_train, "val": trainer.history_val},
        "preds": {
            "pred_best": cell_prob_best,
            "pred_best_adjusted": cell_prob_best_adjusted,
        },
    }

    info_dir = os.path.join(out_dir, "info.pickle")
    logger.info(f"Saving objects to {info_dir}...")
    with open(info_dir, "wb") as f:
        pickle.dump(model_info, f)

    # Extract and save statistics
    logger.info("Extracting and saving statistics...")

    # Load seg_dict once if needed for GeoJSON export
    seg_dict_raw = None
    if save_geojson and json_path is not None:
        with open(json_path, "r") as f:
            seg_dict_raw = json.load(f)
    elif save_geojson and json_path is None:
        logger.warning("save_geojson=True but json_path is None — GeoJSON export will be skipped.")

    # Instantiate PredAnalyzer objects (with seg_dict if available)
    analyzer_best = PredAnalyzer(model_info=model_info, adjusted=False, seg_dict=seg_dict_raw)
    analyzer_best_adj = PredAnalyzer(model_info=model_info, adjusted=True, seg_dict=seg_dict_raw)

    # Extract stats
    stats_best_predicted = analyzer_best.extract_stats(metric="predicted")
    stats_best_all = analyzer_best.extract_stats(metric="all")
    stats_best_adj_predicted = analyzer_best_adj.extract_stats(metric="predicted")
    stats_best_adj_all = analyzer_best_adj.extract_stats(metric="all")

    with pd.ExcelWriter(os.path.join(out_dir, "stats.xlsx")) as writer:
        stats_best_predicted.to_excel(writer, sheet_name="best_predicted", index=False)
        stats_best_all.to_excel(writer, sheet_name="best_all", index=False)
        stats_best_adj_predicted.to_excel(writer, sheet_name="best_adj_predicted", index=False)
        stats_best_adj_all.to_excel(writer, sheet_name="best_adj_all", index=False)

    # GeoJSON export
    if save_geojson and seg_dict_raw is not None:
        import yaml
        from hedest.utils import seg_dict_to_geojson, generate_color_dict

        if color_dict_file is not None:
            with open(color_dict_file, "r") as f:
                color_dict = yaml.safe_load(f)
        else:
            color_dict = generate_color_dict(ct_list, format="special")
            auto_color_path = os.path.join(out_dir, "auto_color_dict.yaml")
            with open(auto_color_path, "w") as f:
                yaml.dump(color_dict, f)
            logger.info(f"Auto-generated color dict saved to {auto_color_path}")

        geojson_path_best = os.path.join(out_dir, "hedest_predictions.geojson")
        geojson_path_best_adj = os.path.join(out_dir, "hedest_predictions_adj.geojson")

        seg_dict_to_geojson(analyzer_best.seg_dict_w_class, geojson_path_best, color_dict=color_dict)
        logger.info(f"GeoJSON (unadjusted) exported to {geojson_path_best}")

        seg_dict_to_geojson(analyzer_best_adj.seg_dict_w_class, geojson_path_best_adj, color_dict=color_dict)
        logger.info(f"GeoJSON (adjusted) exported to {geojson_path_best_adj}")

    logger.info("Secondary deconvolution process completed successfully.")
    logger.info(f"Training time: {TRAIN_TIME}")
