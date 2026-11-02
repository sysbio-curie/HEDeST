"""
Colocalization of cell types, from the proportions and from the HEDeST cells.

- From the proportions (what is done with a deconvolution): the **co-occurrence**, how much more B
  there is in the spots holding A than overall, ``mean(pA pB) / (mean(pA) mean(pB))``. It is the
  share of B among the neighbours of an A cell if every cell of a spot neighboured every other.
- From the HEDeST cells (what the proportions cannot give): the **enrichment at contact**, among
  the cells **touching** an A cell (:mod:`lungstudy.contacts`: nuclei grown by up to 5 µm without
  overlap, sharing a border), the share of B, divided by the share of B among all cells. 1 (log2
  0) means B touches A as often as random mixing would make it; above 1, B gathers around A;
  below 1, B avoids A.

Both are the co-occurrence score of Squidpy (Palla et al. 2022), P(B | near an A) / P(B), with
two definitions of "near": in the same 55 µm spot (assuming every spot holds as many cells), or
touching. ``pair_scores`` puts them side by side for every pair, with 95% intervals from a
spatial block bootstrap, and ``zone`` says whether HEDeST finds more, less or the opposite
colocalization than the deconvolution; ``example_windows`` finds the places that show it.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from lungstudy.contacts import neighbour_counts
from lungstudy.data import SampleData
from lungstudy.data import Window
from lungstudy.views import in_window
from scipy.spatial import cKDTree

PSEUDO = 0.5  # pseudo-count on observed and expected neighbour counts, so rare pairs stay finite
BAND = 0.5  # |HEDeST − deconvolution| below this (log2, i.e. within ×1.41) counts as the same answer

ZONES = {
    "same": "same answer",
    "more": "HEDeST: more colocalization than the spots say",
    "less": "HEDeST: less colocalization than the spots say",
    "flip_apart": "opposite: spots say together, cells say apart",
    "flip_together": "opposite: spots say apart, cells say together",
}


def analysed_types(data: SampleData, min_cells: int = 100) -> List[str]:
    """Every type with at least ``min_cells`` HEDeST cells and a non-zero deconvolution."""

    counts = data.cells["label"].value_counts()
    return [c for c in data.ct if counts.get(c, 0) >= min_cells and data.props[c].mean() > 0]


def _contact_counts(data: SampleData, types: Sequence[str], labels: np.ndarray) -> Tuple:
    """Per cell: number of cells it touches, and number of each type among them."""

    codes = pd.Categorical(labels, categories=list(types)).codes
    return neighbour_counts(data.contacts, np.asarray(codes), len(types))


def pair_scores(
    data: SampleData,
    types: Sequence[str],
    tile_um: float = 500.0,
    n_boot: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """
    For every pair of different types: the deconvolution co-occurrence and the HEDeST
    enrichment at contact, both log2 on the same scale, their gap, and
    95% intervals from a spatial block bootstrap (the slide cut in ``tile_um`` squares, drawn with
    replacement, the spots and the cells of a square drawn together).
    """

    names = data.cells["label"].to_numpy()
    xy = data.cells[["x_um", "y_um"]].to_numpy()
    k = len(types)

    def tile_of(points: np.ndarray) -> np.ndarray:
        return np.floor(points[:, 0] / tile_um).astype(np.int64) * 1_000_000 + np.floor(points[:, 1] / tile_um).astype(
            np.int64
        )

    tiles, cell_tile = np.unique(tile_of(xy), return_inverse=True)
    n_tiles = len(tiles)

    # cells: per tile, sums that the enrichment is made of
    n_all, counts = _contact_counts(data, types, names)
    observed = np.zeros((k, k, n_tiles))
    neighbours = np.zeros((k, n_tiles))
    members = np.zeros((k, n_tiles))
    for a, name in enumerate(types):
        centre = names == name
        neighbours[a] = np.bincount(cell_tile[centre], weights=n_all[centre], minlength=n_tiles)
        members[a] = np.bincount(cell_tile[centre], minlength=n_tiles)
        for b in range(k):
            observed[a, b] = np.bincount(cell_tile[centre], weights=counts[b][centre], minlength=n_tiles)
    all_cells = np.bincount(cell_tile, minlength=n_tiles).astype(float)

    # spots: per tile, sums that the co-occurrence is made of
    spot_xy = data.spots.loc[data.props.index, ["x_um", "y_um"]].to_numpy()
    position = np.searchsorted(tiles, tile_of(spot_xy))
    known = (position < n_tiles) & (tiles[np.minimum(position, n_tiles - 1)] == tile_of(spot_xy))
    p = data.props[list(types)].to_numpy()[known]
    position = position[known]
    joint = np.zeros((k, k, n_tiles))
    marginal = np.zeros((k, n_tiles))
    n_spots = np.bincount(position, minlength=n_tiles).astype(float)
    for a in range(k):
        marginal[a] = np.bincount(position, weights=p[:, a], minlength=n_tiles)
        for b in range(k):
            joint[a, b] = np.bincount(position, weights=p[:, a] * p[:, b], minlength=n_tiles)

    def scores(weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        share = (members @ weight) / (all_cells @ weight)
        obs = observed @ weight
        expected = (neighbours @ weight)[:, None] * share[None, :]
        enrichment = np.log2((obs + PSEUDO) / (expected + PSEUDO))
        enrichment = (enrichment + enrichment.T) / 2
        spots = n_spots @ weight
        mean = (marginal @ weight) / spots
        with np.errstate(divide="ignore", invalid="ignore"):
            cooc = np.log2(((joint @ weight) / spots) / np.outer(mean, mean))
        return enrichment, cooc, (obs + obs.T) / 2, (expected + expected.T) / 2

    contact, cooc, obs, expected = scores(np.ones(n_tiles))
    rng = np.random.default_rng(seed)
    boot_contact, boot_cooc = [], []
    for _ in range(n_boot):
        weight = np.bincount(rng.integers(0, n_tiles, n_tiles), minlength=n_tiles).astype(float)
        e, c, _, _ = scores(weight)
        boot_contact.append(e)
        boot_cooc.append(c)
    boot_contact, boot_cooc = np.asarray(boot_contact), np.asarray(boot_cooc)
    boot_gap = boot_contact - boot_cooc

    rows = []
    for a in range(k):
        for b in range(a + 1, k):
            rows.append(
                {
                    "sample": data.name,
                    "A": types[a],
                    "B": types[b],
                    "pair": f"{types[a]} – {types[b]}",
                    "deconv_cooc": cooc[a, b],
                    "deconv_cooc_low": np.nanquantile(boot_cooc[:, a, b], 0.025),
                    "deconv_cooc_high": np.nanquantile(boot_cooc[:, a, b], 0.975),
                    "contact": contact[a, b],
                    "contact_low": np.quantile(boot_contact[:, a, b], 0.025),
                    "contact_high": np.quantile(boot_contact[:, a, b], 0.975),
                    "gap": contact[a, b] - cooc[a, b],
                    "gap_low": np.nanquantile(boot_gap[:, a, b], 0.025),
                    "gap_high": np.nanquantile(boot_gap[:, a, b], 0.975),
                    "contacts_observed": obs[a, b],
                    "contacts_expected": expected[a, b],
                }
            )
    table = pd.DataFrame(rows)
    table["zone"] = [zone(r) for _, r in table.iterrows()]
    table["robust"] = [robust(r) for _, r in table.iterrows()]

    return table


def zone(row: pd.Series, band: float = BAND) -> str:
    """
    Which of the zones of :data:`ZONES` a pair falls in. 'Opposite' needs both signs to hold
    over the 95% intervals; otherwise a pair off the band is only 'more' or 'less'.
    """

    x, y = row["deconv_cooc"], row["contact"]
    if abs(y - x) < band:
        return "same"
    if x > 0 and y < 0 and row["deconv_cooc_low"] > 0 and row["contact_high"] < 0:
        return "flip_apart"
    if x < 0 and y > 0 and row["deconv_cooc_high"] < 0 and row["contact_low"] > 0:
        return "flip_together"
    return "more" if y > x else "less"


def robust(row: pd.Series) -> bool:
    """For a pair off the band: whether the 95% interval of the gap excludes 0."""

    if row["zone"] == "same":
        return True
    return bool(row["gap_low"] > 0 or row["gap_high"] < 0)


def contact_segments(data: SampleData, a: str, b: str, window: Window) -> np.ndarray:
    """The touching A–B pairs of a window, as segments between nucleus centres (level-0 pixels)."""

    xy_px = data.cells[["x", "y"]].to_numpy()
    labels = data.cells["label"].to_numpy()
    pairs = data.contacts
    first, second = labels[pairs[:, 0]], labels[pairs[:, 1]]
    kind = ((first == a) & (second == b)) | ((first == b) & (second == a))
    pairs = pairs[kind]
    inside = in_window(xy_px[pairs[:, 0]], window, 30) | in_window(xy_px[pairs[:, 1]], window, 30)
    pairs = pairs[inside]

    return np.stack([xy_px[pairs[:, 0]], xy_px[pairs[:, 1]]], axis=1) if len(pairs) else np.zeros((0, 2, 2))


def window_enrichment(data: SampleData, a: str, b: str, window: Window) -> Dict[str, float]:
    """A–B contacts among the cells of a window, against random mixing of these cells."""

    xy_px = data.cells[["x", "y"]].to_numpy()
    inside = in_window(xy_px, window)
    labels = data.cells["label"].to_numpy()
    if inside.sum() < 10:
        return {"observed": 0, "expected": 0.0, "log2": np.nan}
    pairs = data.contacts
    pairs = pairs[inside[pairs[:, 0]] & inside[pairs[:, 1]]]
    kinds = labels[pairs]
    observed = int((((kinds[:, 0] == a) & (kinds[:, 1] == b)) | ((kinds[:, 0] == b) & (kinds[:, 1] == a))).sum())
    names = labels[inside]
    n = len(names)
    share_a, share_b = np.mean(names == a), np.mean(names == b)
    expected = len(pairs) * 2 * share_a * share_b * n / max(n - 1, 1)

    return {
        "observed": observed,
        "expected": float(expected),
        "log2": float(np.log2((observed + PSEUDO) / (expected + PSEUDO))),
    }


def example_windows(
    data: SampleData,
    a: str,
    b: str,
    together: bool,
    n: int = 2,
    size_um: float = 220.0,
    min_cells: int = 20,
    min_proportion: float = 0.15,
    separation_um: float = 1000.0,
    max_candidates: int = 3000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Places that show a pair's cell-level pattern best.

    ``together=False`` (the cells avoid each other): windows centred on the spots where the
    deconvolution gives both types at least ``min_proportion``, ranked by the contacts missing
    compared with random mixing (expected − observed). ``together=True``: windows centred on
    touching A–B pairs, ranked by the contacts in excess (observed − expected). Both need
    ``min_cells`` of each type in the window, and the windows are ``separation_um`` apart.
    """

    props = data.props
    if together:
        labels = data.cells["label"].to_numpy()
        xy = data.cells[["x", "y"]].to_numpy()
        pairs = data.contacts
        kinds = labels[pairs]
        pairs = pairs[((kinds[:, 0] == a) & (kinds[:, 1] == b)) | ((kinds[:, 0] == b) & (kinds[:, 1] == a))]
        middles = (xy[pairs[:, 0]] + xy[pairs[:, 1]]) / 2
        if len(middles) > max_candidates:
            middles = middles[np.random.default_rng(seed).choice(len(middles), max_candidates, replace=False)]
        centres = [(float(x), float(y)) for x, y in middles]
    else:
        spots = props.index[(props[a] >= min_proportion) & (props[b] >= min_proportion)]
        centres = [data.spot_xy(spot) for spot in spots]
    names = data.cells["label"].to_numpy()
    cell_xy = data.cells[["x", "y"]].to_numpy()
    spot_tree = cKDTree(data.spots.loc[props.index, ["x", "y"]].to_numpy())
    rows = []
    for x, y in centres:
        window = data.window(x, y, size_um)
        inside = names[in_window(cell_xy, window)]
        if (inside == a).sum() < min_cells or (inside == b).sum() < min_cells:
            continue
        stats = window_enrichment(data, a, b, window)
        spot = props.index[spot_tree.query([x, y])[1]]
        rows.append({"spot": spot, "x": x, "y": y, "excess": stats["observed"] - stats["expected"], **stats})
    table = pd.DataFrame(rows)
    if not len(table):
        return table
    table = table.sort_values("excess", ascending=not together)
    chosen = []
    for _, row in table.iterrows():
        if all(np.hypot(row["x"] - c["x"], row["y"] - c["y"]) * data.mpp >= separation_um for c in chosen):
            chosen.append(row)
        if len(chosen) == n:
            break

    return pd.DataFrame(chosen).reset_index(drop=True)
