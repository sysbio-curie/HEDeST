from __future__ import annotations

import logging
from typing import Literal
from typing import Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from ._superres_backend.superres_utils import DINOv2_superres_deconv

logger = logging.getLogger(__name__)


def superres_core(
    deconv_adata: ad.AnnData,
    img_dir: str,
    neighb: int = 3,
    radius: int = 129,
    class_weights=None,
    learning_rate=1e-4,
    local_path="~/.panospace_cache/dinov2-base",
    pretrained_model_name="facebook/dinov2-base",
    cache_dir="~/.panospace_cache",
    epoch: int = 50,
    batch_size: int = 32,
    num_workers: int = 0,
    accelerator: Literal["cpu", "gpu"] = "gpu",
    precision: Literal["16-mixed", "32", "64"] = "16-mixed",
    devices: Union[int, str] = "auto",
    seed: int = 42,
    patience: Union[int, None] = None,
    val_frac: float = 0.15,
    mask_mode: Literal["largest", "spots", "none"] = "largest",
    mask_downscale: Union[int, str] = 1,
    mask_min_spots: int = 1,
):
    sr_inferencer = DINOv2_superres_deconv(
        deconv_adata,
        img_dir=img_dir,
        radius=radius,
        neighb=neighb,
        class_weights=class_weights,
        learning_rate=learning_rate,
        local_path=local_path,
        pretrained_model_name=pretrained_model_name,
        cache_dir=cache_dir,
        mask_mode=mask_mode,
        mask_downscale=mask_downscale,
        mask_min_spots=mask_min_spots,
        seed=seed,
    )

    if sr_inferencer.train:
        logger.info("Start training super-resolution model...")
        sr_inferencer.run_train(
            epoch=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
            accelerator=accelerator,
            precision=precision,
            devices=devices,
            seed=seed,
            patience=patience,
            val_frac=val_frac,
        )
    else:
        logger.info("Using pre-trained super-resolution model...")
    sr_adata = sr_inferencer.run_superres(accelerator=accelerator, precision=precision)

    return sr_adata
