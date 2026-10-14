"""
Colocalization of cell types, from the proportions and from the HEDeST cells.

From the proportions (what is done with a deconvolution):

- ``spot_pearson``: Pearson correlation of the proportions of A and B across spots, the
  usual colocalization score; also on centred log-ratios (``clr``), which removes the
  negative bias of proportions that sum to one;
- ``spot_cooccurrence``: how much more B there is in the spots holding A than overall,
  ``mean(pA pB) / (mean(pA) mean(pB))``. It is the share of B among the neighbours of an A
  cell if every cell of a spot neighboured every other, so it lives on the same scale as the
  cell-level score below, at the scale of a spot.

From the HEDeST cells (what the proportions cannot give):

- ``cell_enrichment``: for each radius r, the share of B among the cells within r of an A
  cell, divided by the share of B among all cells. 1 (log2 0) means B is as common around A
  as anywhere; above 1, B gathers around A; below 1, B avoids A. Computed at radii from
  contact (10-15 µm) to several spots (200 µm).
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations_with_replacement
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from lungstudy.data import SampleData
from scipy.spatial import cKDTree

RADII_UM = [10, 15, 25, 35, 50, 75, 100, 150, 200]


def main_types(data: SampleData, min_share: float = 0.01) -> List[str]:
    """The types holding at least ``min_share`` of the deconvolution and of the HEDeST cells."""

    hedest = data.cells["label"].value_counts(normalize=True)
    return [c for c in data.ct if data.props[c].mean() >= min_share and hedest.get(c, 0) >= min_share]


def clr(props: pd.DataFrame, pseudo: float = 0.01) -> pd.DataFrame:
    logs = np.log(props + pseudo)
    return logs.sub(logs.mean(axis=1), axis=0)


def spot_pearson(props: pd.DataFrame, types: Sequence[str], transform: str = "none") -> pd.DataFrame:
    frame = props[list(types)]
    if transform == "clr":
        frame = clr(props)[list(types)]
    return frame.corr()


def spot_cooccurrence(props: pd.DataFrame, types: Sequence[str]) -> pd.DataFrame:
    """log2 of mean(pA pB) / (mean(pA) mean(pB)) across spots."""

    p = props[list(types)].to_numpy()
    joint = p.T @ p / len(p)
    marginal = p.mean(axis=0)
    ratio = joint / np.outer(marginal, marginal)

    return pd.DataFrame(np.log2(ratio), index=list(types), columns=list(types))


def cell_enrichment(
    data: SampleData, types: Sequence[str], radii_um: Sequence[float] = RADII_UM, labels: pd.Series = None
) -> Dict[float, pd.DataFrame]:
    """
    log2 neighbourhood enrichment of every pair of types at every radius (rows: the centre
    cell's type, columns: the neighbour's type).
    """

    labels = data.cells["label"] if labels is None else labels.reindex(data.cells.index)
    xy = data.cells[["x_um", "y_um"]].to_numpy()
    names = labels.to_numpy()
    tree_all = cKDTree(xy)
    trees = {t: cKDTree(xy[names == t]) for t in types}
    share = {t: float(np.mean(names == t)) for t in types}

    out = {}
    for radius in radii_um:
        n_all = tree_all.query_ball_point(xy, radius, return_length=True, workers=-1) - 1
        counts = {}
        for t in types:
            counts[t] = trees[t].query_ball_point(xy, radius, return_length=True, workers=-1) - (names == t)
        matrix = pd.DataFrame(np.nan, index=list(types), columns=list(types))
        for a in types:
            centre = names == a
            total = n_all[centre].sum()
            if total == 0:
                continue
            for b in types:
                matrix.loc[a, b] = np.log2(max(counts[b][centre].sum() / total / share[b], 1e-6))
        out[radius] = matrix

    return out


def symmetric(matrix: pd.DataFrame) -> pd.DataFrame:
    return (matrix + matrix.T) / 2


def pair_table(
    sample: str,
    types: Sequence[str],
    deconv_pearson: pd.DataFrame,
    deconv_pearson_clr: pd.DataFrame,
    deconv_cooc: pd.DataFrame,
    hedest_spot_cooc: pd.DataFrame,
    enrichment: Dict[float, pd.DataFrame],
) -> pd.DataFrame:
    """One row per unordered pair of types (self pairs included) with every score."""

    rows = []
    for a, b in combinations_with_replacement(types, 2):
        row = {
            "sample": sample,
            "A": a,
            "B": b,
            "pair": a if a == b else f"{a} – {b}",
            "self": a == b,
            "deconv_pearson": deconv_pearson.loc[a, b] if a != b else np.nan,
            "deconv_pearson_clr": deconv_pearson_clr.loc[a, b] if a != b else np.nan,
            "deconv_cooc": deconv_cooc.loc[a, b],
            "hedest_spot_cooc": hedest_spot_cooc.loc[a, b],
        }
        for radius, matrix in enrichment.items():
            row[f"cell_{radius:g}um"] = symmetric(matrix).loc[a, b]
        rows.append(row)

    return pd.DataFrame(rows)


def patch_radius(curve: pd.Series) -> float:
    """
    The radius at which the self-enrichment of a type has lost half its contact value: a
    rough size of the patches it forms (below a spot, the deconvolution cannot see it).
    """

    values = curve.to_numpy()
    radii = np.asarray(curve.index, dtype=float)
    if values[0] <= 0:
        return np.nan
    half = values[0] / 2
    below = np.flatnonzero(values <= half)
    if not len(below):
        return np.inf
    i = below[0]
    if i == 0:
        return radii[0]
    # linear interpolation between the two radii around the half value
    return float(radii[i - 1] + (half - values[i - 1]) * (radii[i] - radii[i - 1]) / (values[i] - values[i - 1]))


def cooccurring_spots(data: SampleData, a: str, b: str, n: int = 3, min_cells: int = 10) -> List[str]:
    """The spots where the deconvolution puts the most of both A and B (largest min(pA, pB))."""

    counts = data.spots["n_cells"].reindex(data.props.index).fillna(0)
    score = np.minimum(data.props[a], data.props[b])[counts >= min_cells]

    return list(score.sort_values(ascending=False).index[:n])
