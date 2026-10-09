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

# The ground truth is produced on the simulation side and only read here, as is the cell-type
# colour code (STHELAR_DIR/palette.py, the repository's single definition).
STHELAR_DIR = os.path.join(REPO, "simulations", "semi_simulations", "STHELAR")
MATCH_DIR = os.path.join(STHELAR_DIR, "matches")

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


# ----------------------------------------------------------------------------------------
# Comparison against the segmentation methods (segmentation_*.py)
# ----------------------------------------------------------------------------------------
# HoVerNet and CellViT type every nucleus they segment, with the PanNuke classes, without
# being given any spot proportions. At level 0 the STHELAR annotation is the same three broad
# categories under other names, so the two segmentations can be read as cell-typing methods
# and put next to the three that do need proportions. This is a separate study: it is level 0
# only, three classes only, and on its own cell set, so its numbers are not comparable with
# the ones in ``summary_{config}.csv``.
#
# ``skin_s4`` is left out: its level 0 carries a fourth category, Melanocyte, which PanNuke
# has no class for.
SEG_DIR = os.path.join(OUT_DIR, "segmentation")
SEG_MATCH_DIR = os.path.join(SEG_DIR, "matches")  # CellViT <-> HoVerNet pairs
SEG_PLOT_DIR = os.path.join(SEG_DIR, "plots")
SEG_LEVEL = "level0"
SEG_SAMPLES = [s for s in SAMPLES if s != "skin_s4"]

# PanNuke -> the level 0 categories. Both epithelial classes map to Epithelial: ``neopla`` is
# neoplastic and ``no-neo`` non-neoplastic epithelium, a distinction the STHELAR annotation
# does not make at level 0. ``nolabe`` (no label) and ``necros`` (dead) have no counterpart
# and their cells are dropped.
PANNUKE_TO_BROAD = {
    "neopla": "Epithelial",
    "no-neo": "Epithelial",
    "inflam": "Immune",
    "connec": "Structural",
}
DROPPED_PANNUKE = ["nolabe", "necros"]
BROAD_TYPES = ["Epithelial", "Immune", "Structural"]

# The five methods of this study. The proportion-based three keep the colours they have
# everywhere else; the two segmentations get a purple family of their own, since blue is
# HEDeST's. Only the raw HEDeST output is used here, and it takes the strong blue because it
# is the only HEDeST variant on the figures.
SEG_METHODS = {
    "HoVerNet": "#9e9ac8",
    "CellViT": "#54278f",
    "HistoCell": METHODS["HistoCell"],
    "PanoSpace": METHODS["PanoSpace"],
    "HEDeST": METHODS["HEDeST + PPSA"],
}
SEG_SEGMENTERS = ["HoVerNet", "CellViT"]  # deterministic: one run, no seeds
SEG_PROPORTION_METHODS = ["HistoCell", "PanoSpace", "HEDeST"]  # the ones given spot proportions


def seg_units(config: str) -> List[str]:
    """
    The samples the segmentation comparison can run on, for one configuration.

    Args:
        config: ``"gt"`` or ``"deconv"``.

    Returns:
        The sample names, in ``SEG_SAMPLES`` order.
    """

    if config not in CONFIGS:
        raise ValueError(f"config must be one of {list(CONFIGS)}, got {config!r}")

    if config == "gt":
        return list(SEG_SAMPLES)

    return [s for s in SEG_SAMPLES if os.path.exists(proportions_path(s, SEG_LEVEL, "deconv"))]


def segmentation_json(sample: str, backend: str) -> str:
    """
    One of the two segmentations of a sample.

    Args:
        sample: The sample name.
        backend: ``"hovernet"`` or ``"cellvit"``.

    Returns:
        The path to the segmentation JSON.
    """

    if backend not in ("hovernet", "cellvit"):
        raise ValueError(f"backend must be 'hovernet' or 'cellvit', got {backend!r}")

    return os.path.join(BENCH_ROOT, sample, f"{backend}.json")


def seg_pairs_path(sample: str) -> str:
    """
    The CellViT <-> HoVerNet pairs of a sample, with the PanNuke type of both.

    Args:
        sample: The sample name.

    Returns:
        The path to ``matches_{sample}.csv``.
    """

    return os.path.join(SEG_MATCH_DIR, f"matches_{sample}.csv")


def seg_meta_path(sample: str) -> str:
    """
    The nucleus counts and PanNuke histograms written beside a sample's pairs.

    It is written *after* the pairs file, so its presence is what says the matching of a
    sample finished: a reader that only checks the CSV can catch an array task mid-write.

    Args:
        sample: The sample name.

    Returns:
        The path to ``matches_{sample}.meta.json``.
    """

    return os.path.join(SEG_MATCH_DIR, f"matches_{sample}.meta.json")


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
    The STHELAR colours of a sample-level: ``{cell type: (r, g, b)}`` in 0-1.

    Defined once, beside the data, in ``simulations/semi_simulations/STHELAR/palette.py``;
    this only adapts the ``"level0"`` spelling the comparison uses to the integer it takes.

    Args:
        sample: The sample name.
        level: The annotation level, as ``"level0"``.

    Returns:
        One colour per cell type of that level.
    """

    import sys

    if STHELAR_DIR not in sys.path:
        sys.path.insert(0, STHELAR_DIR)
    from palette import level_palette

    return level_palette(sample, int(level.removeprefix("level")), bench_root=BENCH_ROOT)
