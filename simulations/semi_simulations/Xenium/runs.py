"""
The Xenium HEDeST predictions, as the gridsearch left them.

HEDeST was already run on the two Xenium semi-simulations during the parameter gridsearch
(``gridsearch/semi_sim.yaml``), so nothing is trained here: this module only reads those
runs back and wraps them in the objects ``hedest.analysis`` expects, so that the figures can
be drawn with the ordinary plotting functions.

Two things make that wrapping necessary rather than optional:

* the gridsearch writes ``predictions.npz`` (``raw``, ``ppsa``, ``classes``) and
  ``metrics.json`` per seed, **not** the ``info.pickle`` that :func:`load_run` reads, so
  :class:`HedestRun` has to be built by hand from the arrays;
* the cell ids live once per unit in ``cell_ids.txt``, beside the combinations, because every
  combination of a unit scores the same cells in the same order.

The combination used is the one the gridsearch elected for H-Optimus-0 features over all
datasets (``gridsearch/results/semi_sim/selected.csv``, "best mean rank over all datasets
without PPSA"). The predictions are the **raw** ones, averaged over the three seeds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[3]
DATA = Path("/cluster/CBIO/data1/lgortana")

#: Where the gridsearch wrote its runs, for the H-Optimus-0 features.
GRIDSEARCH = REPO / "models" / "gridsearch_semi_sim" / "hoptimus0"

#: The combination elected in ``gridsearch/results/semi_sim/selected.csv``.
COMBINATION = "norm1_do0_a0.01_lr0.0001_l2_hd1024-512"

#: The gridsearch level of the Xenium datasets: their annotation is not hierarchical.
LEVEL = "single"


@dataclass(frozen=True)
class Sample:
    """
    One Xenium semi-simulation.

    Attributes:
        key: Short name used in file names.
        label: Name shown on the figures.
        dataset: The dataset name the gridsearch used.
        root: Directory holding the simulated spots and the segmentation.
        name: Sample name, which is also the key under ``adata.uns['spatial']``.
        image_dict: Nucleus crops, keyed by cell id.
        rotate: Whether to draw the slide turned a quarter turn. The lung slide is a tall
            narrow strip, so it only fits a row of panels on its side.
    """

    key: str
    label: str
    dataset: str
    root: Path
    name: str
    image_dict: Path
    rotate: bool

    @property
    def proportions_path(self) -> Path:
        return self.root / f"sim_{self.name}_prop_real.csv"

    @property
    def spot_dict_path(self) -> Path:
        return self.root / "spot_dict_real.json"

    @property
    def adata_path(self) -> Path:
        return self.root / "pseudo_adata_real.h5ad"

    @property
    def segmentation_path(self) -> Path:
        return self.root / "pannuke_fast_mask_lvl3_annotated.json"

    @property
    def unit_dir(self) -> Path:
        return GRIDSEARCH / self.dataset / LEVEL


SAMPLES: Dict[str, Sample] = {
    "lung": Sample(
        key="lung",
        label="Xenium Lung",
        dataset="xenium_lung",
        root=DATA / "Xenium_V1_humanLung_Cancer_FFPE" / "sim" / "LuCA",
        name="Xenium_V1_humanLung_Cancer_FFPE",
        image_dict=DATA / "Xenium_V1_humanLung_Cancer_FFPE" / "image_dict_64px_20um.pt",
        rotate=True,
    ),
    "breast": Sample(
        key="breast",
        label="Xenium Breast",
        dataset="xenium_breast",
        root=DATA / "Xenium_FFPE_Human_Breast_Cancer_Rep1" / "sim",
        name="Xenium_FFPE_Human_Breast_Cancer_Rep1",
        image_dict=DATA / "Xenium_FFPE_Human_Breast_Cancer_Rep1" / "image_dict_64px_20um.pt",
        rotate=False,
    ),
}


def cell_ids(sample: Sample) -> List[str]:
    """
    The cells the gridsearch scored, in the order its arrays use.

    Args:
        sample: The sample.

    Returns:
        The cell ids.
    """

    path = sample.unit_dir / "cell_ids.txt"
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def mean_raw_predictions(sample: Sample) -> pd.DataFrame:
    """
    The raw predictions of the elected combination, averaged over its seeds.

    The seeds are averaged *before* anything is read off them, which is what the aggregated
    prediction of a multi-seed run is: see ``hedest.analysis.seeds.SeedEnsemble.to_run``.
    Only ``raw`` is read; the ``ppsa`` array in the same file is left alone.

    Args:
        sample: The sample.

    Returns:
        Cells x cell types probabilities.

    Raises:
        FileNotFoundError: If the combination has no seed with predictions.
    """

    combination = sample.unit_dir / COMBINATION
    seeds = sorted(combination.glob("seed_*"))
    stack, classes = [], None

    for seed_dir in seeds:
        path = seed_dir / "predictions.npz"
        if not path.exists():
            logger.warning(f"{path} is missing and is skipped.")
            continue
        payload = np.load(path, allow_pickle=True)
        if classes is None:
            classes = [str(name) for name in payload["classes"]]
        stack.append(payload["raw"].astype("float64"))

    if not stack:
        raise FileNotFoundError(f"No seed with predictions.npz under {combination}.")

    logger.info(f"{sample.label}: averaging the raw predictions of {len(stack)} seeds of {COMBINATION}.")
    cells = cell_ids(sample)
    mean = np.mean(stack, axis=0)

    if mean.shape[0] != len(cells):
        raise ValueError(f"{sample.label}: {mean.shape[0]} rows of predictions for {len(cells)} cell ids.")

    return pd.DataFrame(mean, index=pd.Index(cells, name="cell_id"), columns=classes)


def load_run(sample: Sample):
    """
    Builds the :class:`~hedest.analysis.loaders.HedestRun` of a sample.

    The run carries the seed-averaged raw predictions and no adjusted ones, so every
    analysis reads the raw table; ``adjusted=False`` makes that explicit rather than relying
    on the fallback.

    Args:
        sample: The sample.

    Returns:
        The run.
    """

    from hedest.analysis.loaders import HedestRun

    predictions = mean_raw_predictions(sample)
    proportions = pd.read_csv(sample.proportions_path, index_col=0)
    proportions.index = proportions.index.map(str)

    with open(sample.spot_dict_path) as handle:
        spot_dict = {str(spot): [str(cell) for cell in cells] for spot, cells in json.load(handle).items()}

    return HedestRun(
        predictions_raw=predictions,
        predictions_adjusted=None,
        spot_dict=spot_dict,
        proportions=proportions,
        params={
            "source": str(sample.unit_dir / COMBINATION),
            "combination": COMBINATION,
            "feature": "hoptimus0",
            "predictions": "raw, mean over seeds",
        },
        n_seeds=len(sorted((sample.unit_dir / COMBINATION).glob("seed_*"))),
        adjusted=False,
    )


def load_analyzer(sample: Sample):
    """
    Builds the :class:`~hedest.analysis.pred_analyzer.PredAnalyzer` of a sample.

    Everything the figures need is attached: the segmentation for the nucleus centroids, the
    AnnData for the spots, and the crops for the galleries.

    Args:
        sample: The sample.

    Returns:
        The analyzer.
    """

    import torch
    from hedest.analysis.pred_analyzer import PredAnalyzer
    from hedest.utils import load_spatial_adata

    run = load_run(sample)
    logger.info(f"{sample.label}: loading the crops of {sample.image_dict.name}.")
    image_dict = torch.load(sample.image_dict, map_location="cpu", weights_only=False)

    return PredAnalyzer(
        run,
        seg=str(sample.segmentation_path),
        adata=load_spatial_adata(str(sample.adata_path)),
        adata_name=sample.name,
        image_dict=image_dict,
        adjusted=False,
    )


__all__ = ["COMBINATION", "SAMPLES", "Sample", "cell_ids", "load_analyzer", "load_run", "mean_raw_predictions"]
