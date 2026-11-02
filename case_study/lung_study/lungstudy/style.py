"""
One colour per cell type for the whole study, shared by the spot pies and the HEDeST cells.

The colours are fixed by name, not by position in a level's list, so a type keeps its colour at
every level (B cells are the same blue at lv3 and lv4) and the pies and the cells of a figure
always match. Lymphocytes get strong colours (B blue, T green, plasma magenta); the abundant
types (epithelial, fibroblasts) get light ones, so that immune cells stand out on top of them.

``merge_t`` folds the T-cell subtypes of lv4 and lv5 into one "T cell": HEDeST does not
separate them (see ``00_settings``), so figures show the T lineage as one colour, in the pies
and in the cells alike.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from hedest.analysis.palette import Palette

COLORS = {
    # lymphocytes
    "B cell": "#1f5fbf",
    "plasma cell": "#e7298a",
    "T cell": "#1a9641",
    "T CD4": "#1a9641",
    "T CD8": "#1a9641",
    "T helper": "#1a9641",
    "T reg": "#1a9641",
    "T CD8 EM": "#1a9641",
    "T CD8 cytotoxic": "#1a9641",
    "T CD8 exhaust": "#1a9641",
    "NK cell": "#17becf",
    "immune_lymphoid": "#1f5fbf",
    # myeloid
    "macrophage": "#ff7f00",
    "LAM macroph": "#ff7f00",
    "intermediate macroph": "#ff7f00",
    "proinflammatory macroph": "#ff7f00",
    "DC": "#ffd92f",
    "mast cell": "#984ea3",
    "immune_myeloid": "#ff7f00",
    "immune": "#1f5fbf",
    # epithelium and stroma
    "epithelial": "#c2b5dd",
    "mesothelial": "#8f8f8f",
    "CAF/fibro": "#c8b23c",
    "CAF/fibro ADH1B+": "#c8b23c",
    "CAF/fibro MYH11+": "#c8b23c",
    "myofibroblast": "#a39021",
    "fibroblast": "#c8b23c",
    "stromal": "#c8b23c",
    "SMC": "#8c510a",
    "perivascular": "#8c510a",
    "endothelial": "#d7301f",
    "perictyes": "#fc9272",
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
