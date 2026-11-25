"""
The samples of the case study, its settings, and where every input and output lives.

All inputs are read from the data folder (``DATA_DIR``) and never modified; every output goes
under ``case_study/lung_study``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

STUDY_DIR = Path(__file__).resolve().parents[1]
HEDEST_REPO = STUDY_DIR.parents[1]
DATA_DIR = Path("/cluster/CBIO/data1/lgortana/LungScape_HEDeST")

CACHE_DIR = STUDY_DIR / "cache"
FIGURE_DIR = STUDY_DIR / "figures"
RESULT_DIR = STUDY_DIR / "results"
QUPATH_DIR = STUDY_DIR / "qupath"

if str(HEDEST_REPO) not in sys.path:
    sys.path.insert(0, str(HEDEST_REPO))

# ------------------------------------------------------------------ samples and settings

SAMPLES = ["mNSCLC_P1", "mNSCLC_P2", "mNSCLC_P3", "mNSCLC_P4", "mNSCLC_P5", "mNSCLC_P6"]
# the slides of the analysis notebooks (01-03); the pre-checks and 00_settings run on all of SAMPLES
ANALYSIS_SAMPLES = ["mNSCLC_P1", "mNSCLC_P2", "mNSCLC_P3"]
ROLES = {
    "mNSCLC_P1": "tumour, B-rich lymphoid aggregates, T cells kept out of the nests",
    "mNSCLC_P2": "immune-rich, lymphoid aggregates and plasma cells",
    "mNSCLC_P3": "tumour with lymphoid aggregates, T cells in the nests",
    "mNSCLC_P4": "tumour with compartmented lymphoid aggregates and a plasma ring",
    "mNSCLC_P5": "tumour with T cells inside the nests, no lymphoid aggregate",
    "mNSCLC_P6": "lymphoid-rich, many aggregates",
}

# The inputs of a slide are one folder of DATA_DIR (anndata/, images/, preprocessing/,
# deconvolution/, hedest_results/), whose files are named after the folder; DATA_DIR also holds
# hedest_summary.csv. The folders are named after the samples, unless the private ID mapping
# (SAMPLE_IDS_FILE, not on git: {"mNSCLC_P1": "<original slide name>", ...}) gives their names.
SAMPLE_IDS_FILE = STUDY_DIR / "sample_ids.json"
SAMPLE_IDS = json.loads(SAMPLE_IDS_FILE.read_text()) if SAMPLE_IDS_FILE.exists() else {}


def use_inputs(data_dir: str | Path = DATA_DIR, id_mapping: bool = True) -> None:
    """
    Reads the inputs from ``data_dir``: with the slide folders named by the private ID mapping
    (``id_mapping``), or named after the samples.
    """

    global DATA_DIR, SAMPLE_IDS
    DATA_DIR = Path(data_dir)
    SAMPLE_IDS = json.loads(SAMPLE_IDS_FILE.read_text()) if id_mapping else {}


def slide_id(name: str) -> str:
    """The name of a sample's folder in DATA_DIR, and of the files in it."""

    return SAMPLE_IDS.get(name, name)


METHOD = "cell2location"
MAIN_LEVEL = "lv3"
# lv3: every analysis; lv4: plasma cells (B and plasma split), and an independent model for the
# B / T zones; lv5: in 00_settings only (lv1 and lv2 are too coarse to say anything about immune cells)
LEVELS = ["lv3", "lv4", "lv5"]
PROPORTIONS = "10pct_filter"  # what the main HEDeST runs are trained on; "raw" for the sensitivity checks
PROPORTION_FILES = {
    "raw": "{sample}_V2_proportions.csv",
    "10pct_filter": "{sample}_V2_proportions_10pct_filter.csv",
}
# the (level, proportions) units cached for every sample
UNITS = [
    ("lv3", "10pct_filter"),
    ("lv3", "raw"),
    ("lv4", "10pct_filter"),
    ("lv4", "raw"),
    ("lv5", "10pct_filter"),
    ("lv5", "raw"),
]
# proportions per level: the 10% filter everywhere except lv5, where it erases most of the 21 types
# (5-13 of them keep a mean of 0.5% after the filter, 20-21 before): the rule is per level, the same
# on every slide
KIND_OF = {"lv2": "10pct_filter", "lv3": "10pct_filter", "lv4": "10pct_filter", "lv5": "raw"}
RUN_NAME = "default"
ADJUSTED = False  # HEDeST without PPSA everywhere (00_settings: PPSA changes 3-11% of the labels, never a subtype)
SEEDS = [0, 1, 2, 3, 4]

SPOT_DIAMETER_UM = 55.0
SPOT_PITCH_UM = 100.0

# HoVer-Net PanNuke classes (external/hovernet/type_info.json)
HOVERNET_TYPES = {
    0: "no label",
    1: "neoplastic",
    2: "inflammatory",
    3: "connective",
    4: "necrosis",
    5: "non-neoplastic epithelial",
}


@dataclass(frozen=True)
class SamplePaths:
    """Where the inputs of one sample live."""

    name: str

    @property
    def stem(self) -> str:
        return slide_id(self.name)

    @property
    def root(self) -> Path:
        return DATA_DIR / self.stem

    @property
    def adata(self) -> Path:
        return self.root / "anndata" / f"{self.stem}_V2_aligned.h5ad"

    @property
    def slide(self) -> Path:
        candidates = sorted((self.root / "images" / "hr_tif").glob("*_hr.tiff"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"{self.name}: expected one *_hr.tiff, found {candidates}")
        return candidates[0]

    @property
    def mpp(self) -> float:
        # The aligned TIFF tags hold a placeholder: the real resolution is in metadata.json.
        with open(self.root / "images" / "metadata.json") as handle:
            return float(json.load(handle)["highres_mpp_x"])

    @property
    def segmentation(self) -> Path:
        return self.root / "preprocessing" / "segmentation" / f"{self.stem}.json"

    @property
    def image_dict(self) -> Path:
        return self.root / "preprocessing" / "image_dict" / f"{self.stem}_64px_20um.pt"

    @property
    def mask(self) -> Path:
        return self.root / "preprocessing" / "mask" / f"{self.stem}.png"

    def proportions(self, level: str = MAIN_LEVEL, kind: str = PROPORTIONS) -> Path:
        return self.root / "deconvolution" / METHOD / level / PROPORTION_FILES[kind].format(sample=self.stem)

    def run_dir(self, level: str = MAIN_LEVEL, kind: str = PROPORTIONS) -> Path:
        return self.root / "hedest_results" / METHOD / level / kind / RUN_NAME

    def cache(self) -> Path:
        return CACHE_DIR / self.name


def sample(name: str) -> SamplePaths:
    """The paths of a sample."""

    return SamplePaths(name)


def figure_path(part: str, name: str, suffix: str = "png") -> Path:
    """Where a figure of a part of the study is written."""

    path = FIGURE_DIR / part / f"{name}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def _rasterize_dense(fig: plt.Figure, max_items: int = 2000) -> None:
    """
    Marks the layers with more than ``max_items`` shapes (cells, spots, pie wedges) as raster,
    so that an SVG stays a few MB: text, axes and outlines stay vectors.
    """

    for ax in fig.axes:
        for collection in ax.collections:
            if max(len(collection.get_offsets()), len(collection.get_paths())) > max_items:
                collection.set_rasterized(True)
        if len(ax.patches) > max_items:
            for patch in ax.patches:
                patch.set_rasterized(True)


def save(fig: plt.Figure, part: str, name: str, dpi: int = 200) -> plt.Figure:
    """
    Writes a figure under ``figures/<part>/<name>.png`` and ``.svg`` and closes it, so a
    notebook shows it once when it is the value of the cell.
    """

    fig.savefig(figure_path(part, name), dpi=dpi, bbox_inches="tight", facecolor="white")
    _rasterize_dense(fig)
    fig.savefig(figure_path(part, name, "svg"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return fig


def result_path(name: str) -> Path:
    """Where a result table is written."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    return RESULT_DIR / name
