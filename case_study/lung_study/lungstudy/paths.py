"""
The samples of the case study, its settings, and where every input and output lives.

All inputs are read from the LungScape HEDeST data folder and never modified; every output
goes under ``case_study/lung_study``.
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

SAMPLES = ["A_AAAA1879", "A_AAAA1638", "A_AAAA1665"]
ROLES = {
    "A_AAAA1879": "tumour with B-rich foci",
    "A_AAAA1638": "immune-rich, lymphoid foci",
    "A_AAAA1665": "tumour with B-rich foci (replicate)",
}

METHOD = "cell2location"
MAIN_LEVEL = "lv3"
LEVELS = ["lv1", "lv2", "lv3", "lv4"]
PROPORTIONS = "10pct_filter"  # what HEDeST is trained on in this study; "raw" is the sensitivity check
PROPORTION_FILES = {
    "raw": "{sample}_V2_proportions.csv",
    "10pct_filter": "{sample}_V2_proportions_10pct_filter.csv",
}
RUN_NAME = "default"
ADJUSTED = False  # HEDeST without PPSA everywhere
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
    """Where the inputs of one LungScape sample live."""

    name: str

    @property
    def root(self) -> Path:
        return DATA_DIR / self.name

    @property
    def adata(self) -> Path:
        return self.root / "anndata" / f"{self.name}_V2_aligned.h5ad"

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
        return self.root / "preprocessing" / "segmentation" / f"{self.name}.json"

    @property
    def image_dict(self) -> Path:
        return self.root / "preprocessing" / "image_dict" / f"{self.name}_64px_20um.pt"

    @property
    def mask(self) -> Path:
        return self.root / "preprocessing" / "mask" / f"{self.name}.png"

    def proportions(self, level: str = MAIN_LEVEL, kind: str = PROPORTIONS) -> Path:
        return self.root / "deconvolution" / METHOD / level / PROPORTION_FILES[kind].format(sample=self.name)

    def run_dir(self, level: str = MAIN_LEVEL, kind: str = PROPORTIONS) -> Path:
        return self.root / "hedest_results" / METHOD / level / kind / RUN_NAME

    def cache(self) -> Path:
        return CACHE_DIR / self.name


def sample(name: str) -> SamplePaths:
    """The paths of a sample."""

    return SamplePaths(name)


def figure_path(part: str, name: str) -> Path:
    """Where a figure of a part of the study is written."""

    path = FIGURE_DIR / part / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def save(fig: plt.Figure, part: str, name: str, dpi: int = 200) -> plt.Figure:
    """
    Writes a figure under ``figures/<part>/<name>.png`` and closes it, so a notebook shows it
    once when it is the value of the cell.
    """

    fig.savefig(figure_path(part, name), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return fig


def result_path(name: str) -> Path:
    """Where a result table is written."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    return RESULT_DIR / name
