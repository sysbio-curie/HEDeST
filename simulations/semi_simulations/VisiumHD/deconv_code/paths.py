"""
The samples of the Visium HD colon cancer deconvolution, and where every input and output lives.

Inputs are read from the sample folders under ``/cluster/CBIO/data1/lgortana`` and never modified:
the pseudo-Visium spots, the references built by ``../reference_code``, and the ground-truth
proportions. Every output goes under ``simulations/semi_simulations/VisiumHD``.

There is one cell2location run per sample, against ``reference_full.h5ad``, whose cell types
refine all three annotation levels; ``aggregate.py`` then sums its columns into each level's
vocabulary. Deconvolving a reference already shaped to a level would need one run per level and
buys nothing: summing one fine posterior makes the levels nest the way the ground truth does,
keeps the rare lineages a level drops from smearing over the types that remain, costs a third of
the GPU time, and lets a vocabulary change be re-applied without refitting anything. The per-level
references ``../reference_code`` also writes are a deliverable in their own right and are not read
here.

The level names follow the proportions files, not the obs columns they came from: ``lv0`` is
``lv1_annot``, ``lv1`` is ``lv2_annot``, ``lv2`` is ``lv3_annot``. Each sample's own
``pseudovisium/sim/levels.json`` is the authority on its vocabulary, and ``metadata.json`` carries
the colour code its figures already use, so nothing here redefines either.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY_DIR = HERE.parent
DATA_ROOT = Path("/cluster/CBIO/data1/lgortana")
SAMPLE_DIR = "Visium_HD_Human_Colon_Cancer_{sample}"

RESULT_DIR = STUDY_DIR / "deconv_results"
FIGURE_DIR = STUDY_DIR / "figures" / "deconv"  # beside figures/hedest, written by ../hedest_code

SAMPLES = ["P1", "P2", "P5"]
LEVELS = ["lv0", "lv1", "lv2"]

REFERENCE = "full"  # the reference deconvolved: its cell types refine every level
CELLTYPE_KEY = "cell_type"
BATCH_KEY = "library"  # the Flex GEM well: the technical unit, unlike patient which is biological
DETECTION_ALPHA = 20.0  # cell2location's recommended value for within-slide technical variability
REFERENCE_EPOCHS = 250
SPATIAL_EPOCHS = 30_000
POSTERIOR_SAMPLES = 1_000


@dataclass(frozen=True)
class Sample:
    """Where one sample's inputs live and what its levels contain."""

    name: str

    @property
    def root(self) -> Path:
        return DATA_ROOT / SAMPLE_DIR.format(sample=self.name)

    @property
    def spots(self) -> Path:
        return self.root / "pseudovisium" / "pseudovisium.h5ad"

    @property
    def metadata(self) -> Path:
        return self.root / "pseudovisium" / "metadata.json"

    @property
    def reference(self) -> Path:
        return self.root / "references" / f"reference_{REFERENCE}.h5ad"

    @property
    def mapping(self) -> Path:
        return self.root / "references" / "mapping.csv"

    def truth(self, level: str) -> Path:
        return self.root / "pseudovisium" / "sim" / f"proportions_{level}.csv"

    @property
    def levels(self) -> dict:
        return json.loads((self.root / "pseudovisium" / "sim" / "levels.json").read_text())

    def vocabulary(self, level: str) -> list[str]:
        """The cell types of one level, in the order its proportions file uses."""

        entry = next(e for e in self.levels["levels"] if f"lv{e['level']}" == level)

        return list(entry["cell_types"])

    def palette(self, level: str) -> dict[str, str]:
        """The colour code this sample's own figures use, so new figures match them."""

        recorded = json.loads(self.metadata.read_text())["seg"]["palette"]
        prefix = f"{level}::"

        return {k[len(prefix) :]: v for k, v in recorded.items() if k.startswith(prefix)}


def sample(name: str) -> Sample:
    return Sample(name)


def run_dir(name: str) -> Path:
    """Where one sample's cell2location run keeps its models and posterior."""

    return RESULT_DIR / "runs" / name


def proportions_path(name: str, level: str) -> Path:
    """Where a sample's proportions at one level are written."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    return RESULT_DIR / f"proportions_{name}_{level}.csv"


def figure_path(name: str) -> Path:
    """Where a figure is written."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    return FIGURE_DIR / f"{name}.png"
