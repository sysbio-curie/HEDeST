"""Single source of truth for the STHELAR method comparison (paths, samples, methods, colours).

Four cell-typing methods are compared on the STHELAR benchmark samples, against the DAPI
ground truth matched by ``simulations/semi_simulations/STHELAR/match_cells.py`` (the semi-
simulated data and its ground truth are built there; everything that *compares* methods
lives here):

* **HistoCell** and **PanoSpace**, already run by ``benchmark/annotation_code/``;
* **HEDeST** and **HEDeST + PPSA**, the raw and adjusted outputs of the same run.

Each method was run under two *configurations*, which differ only in the spot proportions it
was given:

* ``gt`` — the true proportions, ``bench_data/{sample}/sim/{level}/proportions.csv``;
* ``deconv`` — proportions estimated by deconvolution,
  ``benchmark/results/PanoSpace/deconv/{sample}/{level}/proportions.csv``.

The ``gt`` configuration covers all 39 sample-levels; the ``deconv`` one only the 27 the
deconvolution produced (no ``lymph_node_s0``, and the finest levels of several samples are
missing), so every ``deconv`` figure is restricted to those.
"""
from __future__ import annotations

import os
from typing import Dict
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))  # benchmark/comparison_code
REPO = os.path.dirname(os.path.dirname(HERE))

BENCH_ROOT = "/cluster/CBIO/data1/lgortana/STHELAR/bench_data"
BENCHMARK_DIR = os.path.join(REPO, "benchmark", "results")
DECONV_PROP_DIR = os.path.join(BENCHMARK_DIR, "PanoSpace", "deconv")

# The cell-type colours of HEDeST-bench, which STHELAR is coloured with everywhere else:
# one hue family per level-0 category (Epithelial blue, Immune green, Structural orange,
# Melanocyte purple), leaves spread within their family, and a category drawn in the mean
# colour of its leaves — so a cell type keeps its colour across levels and across figures.
BENCH_SRC = "/cluster/CBIO/home/lgortana/HEDeST-bench/src"

# The ground truth is produced on the simulation side and only read here.
MATCH_DIR = os.path.join(REPO, "simulations", "semi_simulations", "STHELAR", "matches")

# Everything this study writes, under benchmark/results/comparison/ (as results/MHAST/).
OUT_DIR = os.path.join(BENCHMARK_DIR, "comparison")
RESULT_DIR = OUT_DIR
PLOT_DIR = os.path.join(OUT_DIR, "plots")
DECONV_INPUT_DIR = os.path.join(OUT_DIR, "deconv_inputs")  # spot dicts filtered to the deconv spots

# HEDeST runs: the gridsearch wrote the gt ones, run_hedest_deconv.sh the deconv ones.
HEDEST_GT_ROOT = os.path.join(REPO, "models", "gridsearch_semi_sim", "hoptimus0")
HEDEST_DECONV_ROOT = os.path.join(REPO, "models", "STHELAR_semi_sim_from_deconv", "hoptimus0")
COMBINATION = "norm1_do0_a0.01_lr0.0001_l2_hd1024-512"  # the HEDeST defaults
SEEDS = [0, 1, 2]

# All 39 sample-levels; the deconv configuration keeps the subset that has proportions.
LEVELS: Dict[str, List[str]] = {
    "breast_s6": ["level0", "level1", "level2", "level3", "level4"],
    "cervix_s0_0": ["level0", "level1", "level2", "level3", "level4"],
    "cervix_s0_1": ["level0", "level1", "level2", "level3"],
    "lung_s3": ["level0", "level1", "level2", "level3", "level4"],
    "lymph_node_s0": ["level0", "level1", "level2", "level3", "level4"],
    "ovary_s1": ["level0", "level1", "level2", "level3", "level4"],
    "prostate_s0": ["level0", "level1", "level2", "level3", "level4"],
    "skin_s4": ["level0", "level1", "level2", "level3", "level4"],
}
SAMPLES = list(LEVELS)

CONFIGS = {"gt": "ground-truth proportions", "deconv": "deconvolution-derived proportions"}

# Display order and colours, shared by every figure.
METHODS = {
    "HistoCell": "#bdbdbd",
    "PanoSpace": "#fd8d3c",
    "HEDeST": "#9ecae1",
    "HEDeST + PPSA": "#08519c",
}
HEDEST_METHODS = {"HEDeST": "raw", "HEDeST + PPSA": "ppsa"}


def units(config: str) -> List[tuple]:
    """
    The (sample, level) pairs a configuration can be evaluated on.

    Args:
        config: ``"gt"`` or ``"deconv"``.

    Returns:
        The pairs, in sample then level order.
    """

    if config not in CONFIGS:
        raise ValueError(f"config must be one of {list(CONFIGS)}, got {config!r}")

    pairs = [(s, lvl) for s in SAMPLES for lvl in LEVELS[s]]
    if config == "gt":
        return pairs

    return [(s, lvl) for s, lvl in pairs if os.path.exists(proportions_path(s, lvl, "deconv"))]


def proportions_path(sample: str, level: str, config: str) -> str:
    """
    The proportion matrix a configuration feeds the methods.

    Args:
        sample: The sample name.
        level: The annotation level.
        config: ``"gt"`` or ``"deconv"``.

    Returns:
        The path.
    """

    if config == "gt":
        return os.path.join(BENCH_ROOT, sample, "sim", level, "proportions.csv")

    return os.path.join(DECONV_PROP_DIR, sample, level, "proportions.csv")


def gt_path(sample: str, level: str) -> str:
    """
    The matched DAPI ground truth of a sample-level.

    Args:
        sample: The sample name.
        level: The annotation level.

    Returns:
        The path to ``gt_{sample}_{level}.json``.
    """

    return os.path.join(MATCH_DIR, f"gt_{sample}_{level}.json")


def spot_dict_path(sample: str, level: str, config: str) -> str:
    """
    The spot dictionary a configuration trains on.

    The deconvolution did not produce proportions for every spot (1 to 289 are missing,
    depending on the sample), so the deconv configuration uses a spot dictionary restricted
    to the spots it did produce; ``prepare_deconv.py`` writes those.

    Args:
        sample: The sample name.
        level: The annotation level.
        config: ``"gt"`` or ``"deconv"``.

    Returns:
        The path.
    """

    if config == "gt":
        return os.path.join(BENCH_ROOT, sample, "spot_dict.json")

    return os.path.join(DECONV_INPUT_DIR, f"spot_dict_{sample}_{level}.json")


def hedest_run_dir(sample: str, level: str, config: str, seed: int) -> str:
    """
    Where one HEDeST run lives.

    Args:
        sample: The sample name.
        level: The annotation level.
        config: ``"gt"`` or ``"deconv"``.
        seed: The seed.

    Returns:
        The run directory, holding ``predictions.npz`` and ``metrics.json``.
    """

    root = HEDEST_GT_ROOT if config == "gt" else HEDEST_DECONV_ROOT

    return os.path.join(root, sample, level, COMBINATION, f"seed_{seed}")


def competitor_dir(method: str, sample: str, level: str, config: str, seed: int) -> str:
    """
    Where one HistoCell or PanoSpace run lives.

    Args:
        method: ``"HistoCell"`` or ``"PanoSpace"``.
        sample: The sample name.
        level: The annotation level.
        config: ``"gt"`` or ``"deconv"``.
        seed: The seed.

    Returns:
        The run directory, holding ``{method}_predictions.csv``.
    """

    folder = "gt" if config == "gt" else "endecon"  # the benchmark calls the deconv config "endecon"

    return os.path.join(BENCHMARK_DIR, method, sample, level, folder, f"seed{seed}")


# Windows of the featured side-by-side figure, in slide coordinates: ((x, y), (w, h)) at
# level 0. The same windows serve both configurations, so the two figures show the same
# tissue.
_PROSTATE_WINDOWS = [
    ((18750, 41250), (1500, 1500)),
    ((11250, 27750), (1500, 1500)),
    ((18750, 33750), (1500, 1500)),
    ((18600, 28200), (1200, 1200)),
    ((15750, 17250), (1500, 1500)),
]

CROP_WINDOWS: Dict[str, dict] = {
    "gt": {"sample": "prostate_s0", "level": "level3", "windows": _PROSTATE_WINDOWS},
    "deconv": {"sample": "prostate_s0", "level": "level3", "windows": _PROSTATE_WINDOWS},
}

# Per-sample crop figures: this many windows of this side, taken where the nuclei are
# densest (see views.dense_windows), the same ones at every level of the sample.
CROP_SIZE = 1500
CROPS_PER_SAMPLE = 3


def bench_palette(sample: str, level: str) -> Dict[str, tuple]:
    """
    The HEDeST-bench colours of a sample-level: ``{cell type: (r, g, b)}`` in 0-1.

    Args:
        sample: The sample name.
        level: The annotation level.

    Returns:
        One colour per cell type of that level.
    """

    import sys

    if BENCH_SRC not in sys.path:
        sys.path.insert(0, BENCH_SRC)
    from hedest_bench.bench import colors as bench_colors

    return bench_colors.level_palette(sample, int(level.removeprefix("level")))
