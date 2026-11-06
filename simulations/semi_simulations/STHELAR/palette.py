"""The STHELAR cell-type colours: one hand-written table, used everywhere in this repository.

Every cell type gets a colour chosen by hand, keyed by category *name*, so a type keeps its
colour across annotation levels, across methods and across every figure here and in
HEDeST-bench. This replaces an earlier generative scheme (family hue + an in-family
hue/lightness ramp, with a parent drawn in the mean colour of its leaves), which produced
muddy in-between parents and hue-identical siblings.

**Six hue regions, one per compartment.** Hue says which family a type belongs to:

    lymphoid        green -> teal           myeloid       light golds
    epithelium      dark warm reds          melanocyte    near-black -> rose magenta
    structural      one blue ladder: stroma indigo in its dark half, vessels azure in
                    its light half, the two interleaved

**Lightness is the second channel, and it is the one that matters.** Under red-green colour
blindness green, gold and red all collapse onto a single yellowish axis, so the lymphoid,
myeloid and epithelial families become hue-identical no matter what is chosen. They are
therefore kept in separate lightness bands -- epithelium dark (L* 28-71), myeloid light
(L* 68-95), lymphoid in between (L* 45-71) -- and each family is itself a legible L* ladder,
because lightness survives every form of colour blindness.

**The two stain colours are out of bounds.** H&E is pink (eosin) and purple
(haematoxylin), and the figures draw cells on a faded H&E background, so a cell type coloured
like the stain is nearly invisible. Nearest stain colour across the palette is dE 16.1.

**Structural is a single blue ladder, and that is a deliberate trade.** Stroma and vessels share
one hue region so that structural never reads as the same family as immune green. It costs
something, and the cost is measured: blue is the stain's own region (haematoxylin dark, faded
haematoxylin pale), and for a red-green colourblind reader blue offers only *one* usable axis --
lightness. Two blue sub-bands separated by *hue* do not work at all: indigo against azure is
dE 3.1 under deuteranopia, because that difference lives almost entirely on the red-green axis.
Hence one interleaved lightness ladder, stroma in the dark half and vessels in the light half,
rather than two hue bands.

Twelve structural names in blue is over-subscribed, and the ceiling is explicit: requiring all
twelve to be mutually distinct (dE >= 10) *and* clear of the stain (dE >= 15) is infeasible;
relaxing distinctness to 8 caps the colourblind floor at 8.6. This table sits at distinctness 9.3
with a stain floor of 16.1, which is why the two closest structural blues are only dE 9.3 apart.
If that becomes a problem in practice, the alternative on record is stroma in olive/sage, which
restores normal-vision separation to 16.8 at the price of sharing a hue region with immune.

**Colours need only be distinct within one annotation level.** No figure draws two levels at
once, so e.g. level-0 ``Structural`` and level-4 ``High_proliferating_melanocyte`` may both be
violet. That slack is what buys wide, genuinely distinguishable sibling bands instead of
shades of one hue.

Checked pairwise over all 39 STHELAR sample-levels, worst closest-pair dE (CIE76, against the
Vienot-Brettel-Mollon simulations) is 12.9 normal / 12.1 deuteranopia / 12.5 protanopia /
3.4 tritanopia, against 13.8 / 7.2 / 5.7 / 4.5 for the scheme this replaces -- so red-green
colour blindness is roughly twice as well served, and normal vision about level.

    from palette import level_palette
    level_palette("lung_s3", 2)        # {category: (r, g, b)} in 0-1

``children()`` below still derives the family -> leaf tree from the proportions files (a
coarse class equals the sum of its children spot by spot, so a leaf can only sit under a
coarse class that is >= it everywhere). The colours no longer need it -- a name is enough --
but ``benchmark/endecon_code/hierarchy.py`` imports it, so it stays.

Two further copies of this table exist on purpose, so that neither consumer has to depend on
this module: ``hedest_bench.celltype_colors`` in HEDeST-bench, and
``external/HistoCell/bench.py`` inside the vendored method. They must agree exactly;
``python -m hedest_bench.check_palette_sync`` in HEDeST-bench asserts it.
"""
from __future__ import annotations

import glob
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd

BENCH_ROOT = "/cluster/CBIO/data1/lgortana/STHELAR/bench_data"

# Grey, reserved for the catch-all label and for anything not in the table.
OTHER_COLOR = (0.63, 0.63, 0.63)
OTHER_LABEL = "Other"

# cell type -> colour. Grouped by compartment, ordered light to dark inside each group.
CELLTYPE_HEX: Dict[str, str] = {
    # --- immune: lymphoid, green -> teal (keeps blue content, survives deuteranopia)
    "Immune": "#1E8B5A",
    "Lymphoid": "#17805C",
    "B_Plasma": "#1B8A4E",
    "B-lineage": "#1B8A4E",
    "B": "#177A46",
    "Plasma": "#4FC2A1",
    "T_NK": "#0D8A86",
    "T/NK-lineage": "#0D8A86",
    "T": "#0F8F88",
    "T_CD8": "#0C7D7C",
    # --- immune: myeloid, light golds (lightness is what separates it from epithelium)
    "Myeloid": "#D49A2E",
    "Monocyte/Macrophage": "#EEBB63",
    "Monocyte/Macrophage/DC": "#E3A944",
    "Mast": "#F7DDA0",
    "DC": "#FCF0D8",
    "Dendritic Cells": "#FCF0D8",
    # --- structural: stroma, the dark half of the shared blue ladder
    "Structural": "#16216A",
    "Stromal": "#1A277F",
    "Fibroblast_Myofibroblast": "#2539B6",
    "CAF": "#3C51D7",
    "Fibroblast": "#7886E3",
    "Fibroblast/Myofibroblast": "#A2ACEC",
    "Myofibroblast": "#101950",
    # --- structural: vessels, the light half of the shared blue ladder
    "Blood_vessel": "#1385BE",
    "Blood vessel": "#1385BE",
    "Blood Vessel": "#1385BE",
    "Endothelial": "#1CA4E9",
    "Angiogenic_endothelial": "#99D6F5",
    "Pericyte": "#53BAEE",
    "Smooth_muscle": "#09415D",
    # --- epithelium: dark warm reds, kept below the myeloid band at every level
    "Epithelial": "#A33A2B",
    "Breast_epithelial_cell": "#A33A2B",
    "Keratinocyte": "#BB4B30",
    "Luminal_epithelial_cell": "#C96A45",
    "Secretory_epithelial_cell": "#C25C3C",
    "Squamous_epithelial_cell": "#78242E",
    "Proliferating_epithelial": "#D4886B",
    "Lung_epithelial_cell": "#A33A2B",
    "Lung_alveolar_typeI_epithelial_cell": "#C26A45",
    "Lung_alveolar_typeII_epithelial_cell": "#78242E",
    "Lung_ciliated_epithelial_cell": "#D9A086",
    # --- melanocyte: near-black plum -> vivid rose magenta
    "Melanocyte": "#2E1424",
    "High_proliferating_melanocyte": "#EB006D",
}


def rgb(name: str) -> Tuple[float, float, float]:
    """
    The colour of a cell type, as floats in 0-1.

    Args:
        name: A cell type. Anything absent from the table -- including the catch-all
            ``"Other"`` -- gets the reserved grey.

    Returns:
        ``(r, g, b)``.
    """

    value = CELLTYPE_HEX.get(name)
    if value is None:
        return OTHER_COLOR

    return tuple(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def unknown(names) -> List[str]:
    """
    The cell types missing from the table, which would silently come out grey.

    Args:
        names: The cell types to check.

    Returns:
        The missing names, excluding the catch-all label.
    """

    return [n for n in names if n not in CELLTYPE_HEX and n != OTHER_LABEL]


def children(coarse: pd.DataFrame, fine: pd.DataFrame, tol: float = 1e-6) -> Dict[str, List[str]]:
    """
    Groups the fine cell types under their parent, from the proportions alone.

    The colours no longer need this -- a cell type is coloured by name -- but
    ``benchmark/endecon_code/hierarchy.py`` uses it to recover the tree.

    Args:
        coarse: The coarse level's ``spots x categories`` proportion matrix.
        fine: The fine level's proportion matrix.
        tol: Tolerance on the "is everywhere >=" test.

    Returns:
        ``{coarse category: [fine categories it contains]}``.
    """

    spots = coarse.index.intersection(fine.index)
    C, F = coarse.loc[spots], fine.loc[spots]
    out: Dict[str, List[str]] = {c: [] for c in coarse.columns}
    for leaf in fine.columns:
        f = F[leaf].to_numpy()
        cand = {
            c: float((C[c].to_numpy() - f).sum()) for c in coarse.columns if bool(((C[c].to_numpy() - f) >= -tol).all())
        }
        out[min(cand, key=cand.get)].append(leaf)

    return out


def levels_of(sample: str, bench_root: Optional[str] = None) -> List[int]:
    """
    The annotation levels of a sample that have a proportion matrix, coarse to fine.

    Args:
        sample: The sample name.
        bench_root: The ``bench_data`` root; ``BENCH_ROOT`` when omitted.

    Returns:
        The level numbers, ascending.
    """

    sim = os.path.join(bench_root or BENCH_ROOT, sample, "sim")

    return sorted(
        int(os.path.basename(d)[5:])
        for d in glob.glob(os.path.join(sim, "level*"))
        if os.path.exists(os.path.join(d, "proportions.csv"))
    )


def _read_level(sample: str, level: int, bench_root: Optional[str] = None) -> pd.DataFrame:
    """Reads one level's proportion matrix, spot ids as strings."""

    path = os.path.join(bench_root or BENCH_ROOT, sample, "sim", f"level{level}", "proportions.csv")

    return pd.read_csv(path, index_col=0).rename(index=str)


def leaf_colors(sample: str, bench_root: Optional[str] = None) -> Dict[str, Tuple[float, float, float]]:
    """
    The colour of every finest-level cell type of a sample.

    Args:
        sample: The sample name.
        bench_root: The ``bench_data`` root; ``BENCH_ROOT`` when omitted.

    Returns:
        ``{leaf cell type: (r, g, b)}`` in 0-1.
    """

    return level_palette(sample, levels_of(sample, bench_root)[-1], bench_root)


def level_palette(sample: str, level: int, bench_root: Optional[str] = None) -> Dict[str, Tuple[float, float, float]]:
    """
    The colour of every cell type of one sample-level.

    Args:
        sample: The sample name.
        level: The annotation level, as an integer.
        bench_root: The ``bench_data`` root; ``BENCH_ROOT`` when omitted.

    Returns:
        ``{category: (r, g, b)}`` in 0-1.

    Raises:
        KeyError: If the level holds a cell type the table does not define, which would
            otherwise be drawn in the grey reserved for "other".
    """

    cats = list(_read_level(sample, level, bench_root).columns)
    missing = unknown(cats)
    if missing:
        raise KeyError(f"{sample} level{level}: cell types absent from the colour table: {missing}")

    return {cat: rgb(cat) for cat in cats}
