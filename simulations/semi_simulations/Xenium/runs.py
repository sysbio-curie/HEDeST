"""
The Xenium HEDeST predictions: the ten seeds of ``hedest_code/xenium.yaml``.

HEDeST is trained on the two Xenium semi-simulations by the gridsearch engine
(``gridsearch/gridsearch.py``) on one combination and ten seeds, into ``hedest_models/``
(``sbatch hedest_code/hedest_array.sh``). This module reads those runs back and wraps them in
the objects ``hedest.analysis`` expects, so that the figures can be drawn with the ordinary
plotting functions.

Two things make that wrapping necessary rather than optional:

* the engine writes ``predictions.npz`` (``raw``, ``ppsa``, ``classes``) and ``metrics.json``
  per seed, **not** the ``info.pickle`` that :func:`load_run` reads, so :class:`HedestRun` has
  to be built by hand from the arrays;
* the cell ids live once per unit in ``cell_ids.txt``, beside the combination, and the
  train/val/test split of a seed is not saved: :func:`train_spot_dict` recomputes it, as the
  engine drew it.

The combination is the one the gridsearch elected for H-Optimus-0 features over all datasets
(``gridsearch/results/semi_sim/selected.csv``, "best mean rank over all datasets without
PPSA"). :func:`load_run` gives the **raw** predictions averaged over the ten seeds, for the
slide figures; :func:`seed_run` gives one seed, raw or PPSA-adjusted, for the performance
figures, which carry the spread between seeds.
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

#: Where ``hedest_code/xenium.yaml`` writes its runs, for the H-Optimus-0 features.
MODELS = Path(__file__).resolve().parent / "hedest_models" / "hoptimus0"

#: The combination elected in ``gridsearch/results/semi_sim/selected.csv``.
COMBINATION = "norm1_do0_a0.01_lr0.0001_l2_hd1024-512"

#: The level of the Xenium datasets in the engine's layout: their annotation is not hierarchical.
LEVEL = "single"


@dataclass(frozen=True)
class Sample:
    """
    One Xenium semi-simulation.

    Attributes:
        key: Short name used in file names.
        label: Name shown on the figures.
        dataset: The dataset name of ``hedest_code/xenium.yaml``.
        root: Directory holding the simulated spots and the segmentation.
        name: Sample name, which is also the key under ``adata.uns['spatial']``.
        image_dict: Nucleus crops, keyed by cell id.
        rotate: Whether to draw the slide turned a quarter turn. The lung slide is a tall
            narrow strip, so it only fits a row of panels on its side.
        colors: One colour per cell type, in the order of the prediction columns (see
            :data:`SAMPLES`).
    """

    key: str
    label: str
    dataset: str
    root: Path
    name: str
    image_dict: Path
    rotate: bool
    colors: Dict[str, str]

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
    def ground_truth_path(self) -> Path:
        return self.root / f"sim_{self.name}_gt.csv"

    @property
    def unit_dir(self) -> Path:
        return MODELS / self.dataset / LEVEL


# The colour code of the project, as the STHELAR benchmark and the two case studies use it: hue
# says which compartment a cell type belongs to and lightness says which member of it, in even
# steps. Lightness carries the load because it is the channel that survives colour blindness --
# under red-green blindness green, gold and red collapse onto one yellowish axis -- so a family is
# a ladder of lightness rather than a cluster of shades:
#
#     lymphoid    one green ladder, B dark / plasma mid / T light
#     myeloid     light gold
#     epithelium  dark warm reds
#     structural  one blue ladder, stroma in its dark half, vessels in its light half
#
# Neither H&E stain colour is used. Myoepithelial cells are the one type with no counterpart
# elsewhere in the project: they take a second, darker epithelial rung.
#
# Checked pairwise over the seven types of each sample, which all share the pies and the legend:
# the closest pair is dE 22.8 on lung (B cell / plasma cell) and 24.0 on breast
# (Myoepithelial_Cells / Tumor_Cells), under normal, deuteranopic and protanopic vision.
#
# The order is the order of the prediction columns, so that ``palette.names == run.ct_list``.
LUNG_COLORS = {
    "T cell": "#67E49D",  # lymphoid, light rung
    "Epithelial cell": "#A33A2B",  # epithelium
    "Macrophage/Monocyte": "#EEBB63",  # myeloid gold
    "Stromal": "#04429F",  # structural, stroma rung
    "Plasma cell": "#20973C",  # lymphoid, middle rung
    "B cell": "#024B13",  # lymphoid, dark rung
    "Endothelial cell": "#51ABFB",  # structural, vessel rung
}
BREAST_COLORS = {
    "Tumor_Cells": "#A33A2B",  # epithelium
    "Macrophages": "#EEBB63",  # myeloid gold
    "Stromal_Cells": "#04429F",  # structural, stroma rung
    "Myoepithelial_Cells": "#701C33",  # epithelium, the darker rung
    "T_Cells": "#67E49D",  # lymphoid, light rung
    "Endothelial_Cells": "#51ABFB",  # structural, vessel rung
    "B_Cells": "#024B13",  # lymphoid, dark rung
}

SAMPLES: Dict[str, Sample] = {
    "lung": Sample(
        key="lung",
        label="Xenium Lung",
        dataset="xenium_lung",
        root=DATA / "Xenium_V1_humanLung_Cancer_FFPE" / "sim" / "LuCA",
        name="Xenium_V1_humanLung_Cancer_FFPE",
        image_dict=DATA / "Xenium_V1_humanLung_Cancer_FFPE" / "image_dict_64px_20um.pt",
        rotate=True,
        colors=LUNG_COLORS,
    ),
    "breast": Sample(
        key="breast",
        label="Xenium Breast",
        dataset="xenium_breast",
        root=DATA / "Xenium_FFPE_Human_Breast_Cancer_Rep1" / "sim",
        name="Xenium_FFPE_Human_Breast_Cancer_Rep1",
        image_dict=DATA / "Xenium_FFPE_Human_Breast_Cancer_Rep1" / "image_dict_64px_20um.pt",
        rotate=False,
        colors=BREAST_COLORS,
    ),
}


def cell_ids(sample: Sample) -> List[str]:
    """
    The cells the runs scored, in the order their arrays use.

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
    stack, classes = [], None

    for seed_dir in seed_dirs(sample):
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


def seed_dirs(sample: Sample) -> List[Path]:
    """
    The finished seeds of the combination, in seed order.

    Args:
        sample: The sample.

    Returns:
        The ``seed_*`` directories holding a ``metrics.json``.
    """

    found = (sample.unit_dir / COMBINATION).glob("seed_*")

    return sorted((path for path in found if (path / "metrics.json").exists()), key=lambda p: int(p.name[5:]))


def spot_dict(sample: Sample) -> Dict[str, List[str]]:
    """
    The spots as the engine reads them: string ids, empty spots dropped.

    The order matters: it is the order :func:`train_spot_dict` splits.

    Args:
        sample: The sample.

    Returns:
        Spot id to the ids of its cells.
    """

    with open(sample.spot_dict_path) as handle:
        spots = {str(spot): [str(cell) for cell in cells] for spot, cells in json.load(handle).items()}

    return {spot: cells for spot, cells in spots.items() if cells}


def train_spot_dict(sample: Sample, seed_dir: Path) -> Dict[str, List[str]]:
    """
    The training spots of one seed, redrawn as the engine drew them.

    The engine does not save its split, but draws it with
    :func:`hedest.dataset_utils.split_data` from the spot order and the seed alone, so the same
    call gives the same spots. The split sizes are read from the run's own ``metrics.json``.

    Args:
        sample: The sample.
        seed_dir: The run.

    Returns:
        Training spot id to the ids of its cells.
    """

    from hedest.dataset_utils import pp_prop
    from hedest.dataset_utils import split_data

    with open(seed_dir / "metrics.json") as handle:
        metrics = json.load(handle)
    train, *_ = split_data(
        spot_dict(sample),
        pp_prop(str(sample.proportions_path)),
        train_size=metrics["params"]["train_size"],
        val_size=metrics["params"]["val_size"],
        rs=metrics["seed"],
    )

    return train


def seed_run(sample: Sample, seed_dir: Path, adjusted: bool):
    """
    One seed as a :class:`~hedest.analysis.loaders.HedestRun`, with its training spots.

    Args:
        sample: The sample.
        seed_dir: The run.
        adjusted: Whether the run reads the PPSA-adjusted predictions rather than the raw ones.

    Returns:
        The run.
    """

    from hedest.analysis.loaders import HedestRun
    from hedest.dataset_utils import pp_prop

    payload = np.load(seed_dir / "predictions.npz", allow_pickle=True)
    cells = pd.Index(cell_ids(sample), name="cell_id")
    classes = [str(name) for name in payload["classes"]]
    raw, ppsa = (pd.DataFrame(payload[key].astype("float64"), index=cells, columns=classes) for key in ("raw", "ppsa"))

    return HedestRun(
        predictions_raw=raw,
        predictions_adjusted=ppsa,
        spot_dict=spot_dict(sample),
        proportions=pp_prop(str(sample.proportions_path)),
        params={"source": str(seed_dir), "combination": COMBINATION, "feature": "hoptimus0"},
        train_spot_dict=train_spot_dict(sample, seed_dir),
        run_dir=seed_dir,
        seeds=[int(seed_dir.name[5:])],
        adjusted=adjusted,
    )


def true_labels(sample: Sample) -> pd.Series:
    """
    The ground truth type of every cell: the argmax of ``sim_*_gt.csv``, as the engine scores it.

    Lung's ground truth is soft for 40% of its cells; breast's is one-hot.

    Args:
        sample: The sample.

    Returns:
        Cell id to cell type.
    """

    truth = pd.read_csv(sample.ground_truth_path, index_col=0)
    truth.index = truth.index.map(str)

    return truth.idxmax(axis=1).rename("cell_type")


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
        n_seeds=len(seed_dirs(sample)),
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
    from hedest.analysis.palette import Palette
    from hedest.analysis.pred_analyzer import PredAnalyzer
    from hedest.utils import load_spatial_adata

    run = load_run(sample)
    if list(sample.colors) != list(run.ct_list):
        raise ValueError(
            f"{sample.label}: the palette and the prediction columns must list the same cell types "
            f"in the same order; palette {list(sample.colors)}, predictions {list(run.ct_list)}."
        )

    logger.info(f"{sample.label}: loading the crops of {sample.image_dict.name}.")
    image_dict = torch.load(sample.image_dict, map_location="cpu", weights_only=False)

    return PredAnalyzer(
        run,
        seg=str(sample.segmentation_path),
        adata=load_spatial_adata(str(sample.adata_path)),
        adata_name=sample.name,
        image_dict=image_dict,
        palette=Palette.from_colors(sample.colors),
        adjusted=False,
    )


__all__ = [
    "BREAST_COLORS",
    "COMBINATION",
    "LUNG_COLORS",
    "SAMPLES",
    "Sample",
    "cell_ids",
    "load_analyzer",
    "load_run",
    "mean_raw_predictions",
    "seed_dirs",
    "seed_run",
    "spot_dict",
    "train_spot_dict",
    "true_labels",
]
