"""
One colour per cell type for the whole study, shared by the spot pies and the HEDeST cells.

Hue says which compartment a type belongs to, and **lightness says which member**, in even steps:

    lymphoid    one green ladder        myeloid      light golds
    epithelium  dark warm reds          structural   one blue ladder

Lightness carries the load because it is the channel that survives colour blindness: under
red-green blindness green, gold and red all collapse onto one yellowish axis, so those families
are separated by L* band rather than by hue. Inside a single hue there is nothing *but* lightness
-- two blue sub-bands split by hue measure dE 3.1 under deuteranopia -- so each family is one
ladder with rungs about 15 L* apart, not a cluster of shades:

    lymphoid    B cell 27, plasma cell 55, T cell 82, with NK cell on a turquoise at 68
    structural  myofibroblast 16, CAF/fibro 30, SMC 45, endothelial 68, pericytes 81

The lymphoid and structural ladders were rebuilt on 2026-11-03: B/plasma/T used to be green,
mint and teal, which reads as three shades of one colour however far apart they measure, and
SMC / endothelial sat at dE 13.0. The three lymphoid rungs are now 27 L* apart and
SMC / endothelial measures 28.3.

Neither H&E stain colour is used, because the figures draw cells over a faded H&E background and
a cell in eosin pink or haematoxylin purple would vanish into it. One limit is worth knowing: a
*light* green cannot be pulled far from faded eosin for a red-green colourblind viewer, since both
wash out to a pale cream. The whole light-green band tops out near dE 11 (measured over hue
130-160 by every lightness and saturation), so T cell sits at 10.9 -- the same trade the pale
golds already make, DC being at 7.5.

Colours are fixed by name, not by position in a level's list, so a type keeps its colour at every
level and the pies and the cells of a figure always match. Checked pairwise over the real type set
of every level, with the T subtypes merged as the figures draw them: the worst closest pair is
dE 12.2 at lv2 and 12.8 at lv3/lv4/lv5, both of them inherited gold-against-salmon pairs
(``immune_myeloid`` or ``macrophage`` against ``mesothelial``); every pair among the colours set
here is 15.8 or more.

A subtype deliberately shares its parent's colour (the three macrophage states, the two CAF/fibro
states, every T subtype): HEDeST does not separate them, so the figures show one colour per
lineage. ``merge_t`` folds the T-cell subtypes of lv4 and lv5 into one "T cell" for the same
reason (see ``00_settings``). Two lv2 parents ride a rung of the ladder their children sit on and
never share a figure with them: ``fibroblast`` takes the CAF/fibro rung, ``perivascular`` the SMC
rung.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from hedest.analysis.palette import Palette

COLORS = {
    # --- immune, lymphoid: one green ladder, dark -> light, and a turquoise for NK
    "B cell": "#024B13",
    "plasma cell": "#20973C",
    "T cell": "#67E49D",
    "T CD4": "#67E49D",
    "T CD8": "#67E49D",
    "T helper": "#67E49D",
    "T reg": "#67E49D",
    "T CD8 EM": "#67E49D",
    "T CD8 cytotoxic": "#67E49D",
    "T CD8 exhaust": "#67E49D",
    "NK cell": "#37B5BE",
    "immune_lymphoid": "#20973C",
    "immune": "#20973C",
    # --- immune, myeloid: light golds
    "macrophage": "#EEBB63",
    "LAM macroph": "#EEBB63",
    "intermediate macroph": "#EEBB63",
    "proinflammatory macroph": "#EEBB63",
    "mast cell": "#F7DDA0",
    "DC": "#FCF0D8",
    "immune_myeloid": "#D49A2E",
    # --- epithelium: dark warm reds
    "epithelial": "#A33A2B",
    "mesothelial": "#D4886B",
    # --- structural, stroma: the dark half of the blue ladder
    "myofibroblast": "#0D274A",
    "stromal": "#04429F",
    "CAF/fibro": "#04429F",
    "CAF/fibro ADH1B+": "#04429F",
    "CAF/fibro MYH11+": "#04429F",
    "fibroblast": "#04429F",
    # --- structural, vessels: the light half of the same ladder
    "SMC": "#3D56FA",
    "perivascular": "#3D56FA",
    "endothelial": "#51ABFB",
    "perictyes": "#AFC9FD",
}
OTHER = "#bdbdbd"

T_SUBTYPES = ["T CD4", "T CD8", "T helper", "T reg", "T CD8 EM", "T CD8 cytotoxic", "T CD8 exhaust"]
MACROPHAGES = ["LAM macroph", "intermediate macroph", "proinflammatory macroph"]
CAF_SUBTYPES = ["CAF/fibro ADH1B+", "CAF/fibro MYH11+"]
# lv4 / lv5 types grouped back to their lv3 parent, for figures and lineage-level comparisons
PARENT = {
    **{t: "T cell" for t in T_SUBTYPES},
    **{m: "macrophage" for m in MACROPHAGES},
    **{c: "CAF/fibro" for c in CAF_SUBTYPES},
}


def palette(names: Sequence[str]) -> Palette:
    """The study palette for a level's cell types (unknown names in grey)."""

    return Palette.from_colors({n: COLORS.get(n, OTHER) for n in names}, names=list(names))


def merge_t(labels: pd.Series) -> pd.Series:
    """Cell labels with every T-cell subtype written "T cell"."""

    return labels.replace({t: "T cell" for t in T_SUBTYPES})


def merge_t_props(props: pd.DataFrame) -> pd.DataFrame:
    """Spot proportions with the T-cell subtypes summed into one "T cell" column."""

    subtypes = [c for c in props.columns if c in T_SUBTYPES]
    if not subtypes:
        return props
    merged = props.drop(columns=subtypes)
    merged.insert(0, "T cell", props[subtypes].sum(axis=1))

    return merged


def lineage(labels: pd.Series) -> pd.Series:
    """Cell labels mapped to their lv3 parent (T subtypes to T cell, macrophage subtypes to macrophage...)."""

    return labels.replace(PARENT)
