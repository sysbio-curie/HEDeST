"""
Immune infiltration of the epithelium: do immune cells enter the epithelial nests, or stay in the
stroma around them?

**The epithelial compartment** is drawn from the HEDeST cells (10 µm grid, 20 µm smoothing,
epithelial cells at least half of the cells, patches of at least 5,000 µm²), and every cell gets
its signed distance to its border (negative inside). This is the line pathologists draw between
intra-epithelial and stromal lymphocytes. There is no tumour annotation: the compartment is
mostly tumour on these slides, but can include normal epithelium.

**Profiles.** The share of each immune type among all the cells, by signed distance
(``infiltration_profile``, which also gives cells per mm² of tissue). The deconvolution
counterpart (``spot_ring_profile``) asks the same question at the spots' resolution, about the
same border: a spot is inside the compartment when its centre is, and lies some rings of spots
from the border (spot to spot); each ring gets the mean proportion of each type over its spots.

**What the spots miss** (``deep_hexagons``): around each spot deep in the epithelium, the T cells
HEDeST finds in its hexagon (the cells closer to it than to any other spot) against the T
proportion of the deconvolution, filtered (what HEDeST was trained on) and raw.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from lungstudy import geometry as G
from lungstudy.data import SampleData
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as label_components

DEEP_UM = -20.0  # "inside the epithelium": deeper than 20 µm from the border


def epithelial_compartment(
    data: SampleData,
    epithelial: str = "epithelial",
    bin_um: float = 10.0,
    sigma_um: float = 20.0,
    min_share: float = 0.5,
    min_area_um2: float = 5000.0,
) -> Dict[str, object]:
    """
    The epithelial compartment as a raster, and the signed distance of every cell to its
    border (µm; negative inside the epithelium, positive in the stroma).
    """

    xy = data.cells[["x_um", "y_um"]].to_numpy()
    origin = xy.min(axis=0) - 5 * bin_um
    shape = tuple((np.ceil((xy.max(axis=0) - origin) / bin_um) + 5).astype(int))[::-1]
    gx = ((xy[:, 0] - origin[0]) / bin_um).astype(int)
    gy = ((xy[:, 1] - origin[1]) / bin_um).astype(int)
    total = np.zeros(shape)
    epi = np.zeros(shape)
    np.add.at(total, (gy, gx), 1)
    is_epi = (data.cells["label"] == epithelial).to_numpy()
    np.add.at(epi, (gy[is_epi], gx[is_epi]), 1)
    sigma = sigma_um / bin_um
    total_s = gaussian_filter(total, sigma)
    epi_s = gaussian_filter(epi, sigma)
    occupied = total_s[total > 0]
    tissue = total_s > 0.25 * np.median(occupied)
    mask = tissue & (epi_s / np.maximum(total_s, 1e-9) >= min_share)
    components, n = label_components(mask)
    if n:
        sizes = np.bincount(components.ravel())
        small = sizes * bin_um**2 < min_area_um2
        small[0] = False
        mask[small[components]] = False

    inside = distance_transform_edt(mask) * bin_um
    outside = distance_transform_edt(~mask) * bin_um
    signed = np.where(mask, -inside, outside)
    cell_signed = pd.Series(signed[gy, gx], index=data.cells.index, name="signed_distance_um")

    return {"mask": mask, "tissue": tissue, "signed": signed, "origin": origin, "bin_um": bin_um, "cells": cell_signed}


def infiltration_profile(
    data: SampleData,
    compartment: Dict[str, object],
    types: Sequence[str],
    edges_um: Sequence[float] = tuple(range(-100, 301, 20)),
    labels: Optional[pd.Series] = None,
    cells: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Share of each type among all the cells, and its density (cells per mm² of tissue), by signed
    distance to the epithelial border.
    ``cells`` restricts the count to some cells (a boolean mask), e.g. those inside spots.
    """

    labels = data.cells["label"] if labels is None else labels.reindex(data.cells.index)
    signed_grid = compartment["signed"]
    tissue = compartment["tissue"]
    pixel_mm2 = (compartment["bin_um"] / 1000) ** 2
    distance = compartment["cells"].to_numpy()
    keep = np.ones(len(distance), dtype=bool) if cells is None else np.asarray(cells)
    edges = np.asarray(edges_um, dtype=float)
    area = np.histogram(signed_grid[tissue], bins=edges)[0] * pixel_mm2
    total = np.histogram(distance[keep], bins=edges)[0]
    rows = []
    for cell_type in types:
        counts = np.histogram(distance[keep & (labels == cell_type).to_numpy()], bins=edges)[0]
        for i in range(len(edges) - 1):
            rows.append(
                {
                    "cell_type": cell_type,
                    "distance_um": (edges[i] + edges[i + 1]) / 2,
                    "density_per_mm2": counts[i] / area[i] if area[i] > 0 else np.nan,
                    "share": counts[i] / total[i] if total[i] > 0 else np.nan,
                    "n": int(counts[i]),
                }
            )

    return pd.DataFrame(rows)


def spots_inside(data: SampleData, compartment: Dict[str, object]) -> pd.Series:
    """True for the spots whose centre lies in the epithelial compartment."""

    spots = data.props.index
    xy = data.spots.loc[spots, ["x_um", "y_um"]].to_numpy()
    mask, origin, bin_um = compartment["mask"], compartment["origin"], compartment["bin_um"]
    gx = ((xy[:, 0] - origin[0]) / bin_um).astype(int)
    gy = ((xy[:, 1] - origin[1]) / bin_um).astype(int)
    valid = (gx >= 0) & (gx < mask.shape[1]) & (gy >= 0) & (gy < mask.shape[0])
    inside = np.zeros(len(spots), dtype=bool)
    inside[valid] = mask[gy[valid], gx[valid]]

    return pd.Series(inside, index=spots)


def spot_ring_profile(
    data: SampleData,
    compartment: Dict[str, object],
    types: Sequence[str],
    max_ring: int = 4,
    props: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    What the deconvolution says about the same border, at its own resolution: every spot is
    placed in signed rings of spots from the HEDeST epithelial border (``G.spot_rings``; -1: inside,
    touching the stroma; 1: outside, touching the epithelium), and each ring gets the mean
    proportion of each type over its spots.
    """

    props = data.props if props is None else props
    rings = G.spot_rings(data, spots_inside(data, compartment).reindex(props.index), signed=True, max_ring=max_ring)
    rows = []
    for cell_type in types:
        for ring, values in props[cell_type].groupby(rings):
            rows.append(
                {"cell_type": cell_type, "ring": int(ring), "proportion": float(values.mean()), "spots": len(values)}
            )

    return pd.DataFrame(rows)


def deep_hexagons(
    data: SampleData,
    compartment: Dict[str, object],
    raw_props: pd.DataFrame,
    t_type: str = "T cell",
    epithelial: str = "epithelial",
    max_depth_um: float = -30.0,
    min_epithelial: float = 0.8,
    min_cells: int = 15,
) -> pd.DataFrame:
    """
    The spots deep in the epithelium (their hexagon's cells at a median depth below
    ``max_depth_um``, the filtered deconvolution ≥ ``min_epithelial`` epithelial, ≥ ``min_cells``
    nuclei in the hexagon), with what each source says about T cells there: HEDeST's T cells in
    the hexagon (and how many lie deeper than 20 µm), the filtered and the raw deconvolution.
    """

    cells = data.cells.assign(distance=compartment["cells"])
    hexagon = data.pitch_px * data.mpp / np.sqrt(3)
    zone = cells[(cells["nearest_spot_dist_um"] <= hexagon) & cells["nearest_spot"].isin(data.props.index)]
    is_t = zone["label"] == t_type
    per = zone.groupby("nearest_spot").agg(n=("label", "size"), depth=("distance", "median"))
    per["T"] = is_t.groupby(zone["nearest_spot"]).sum().reindex(per.index).fillna(0).astype(int)
    per["T_deep"] = (
        (is_t & (zone["distance"] < DEEP_UM))
        .groupby(zone["nearest_spot"])
        .sum()
        .reindex(per.index)
        .fillna(0)
        .astype(int)
    )
    per["deconv_T"] = data.props.loc[per.index, t_type]
    per["deconv_epithelial"] = data.props.loc[per.index, epithelial]
    per["deconv_T_raw"] = raw_props.loc[per.index, t_type]
    per["deconv_immune_raw"] = raw_props.loc[
        per.index,
        [c for c in raw_props.columns if c in ("T cell", "B cell", "NK cell", "macrophage", "DC", "mast cell")],
    ].sum(axis=1)
    per["hedest_T_share"] = per["T"] / per["n"]

    return per[(per["depth"] < max_depth_um) & (per["deconv_epithelial"] >= min_epithelial) & (per["n"] >= min_cells)]


def draw_border(ax, data: SampleData, compartment: Dict[str, object], color="#d6604d", linewidth: float = 1.5) -> None:
    """Draws the border of the epithelial compartment (level-0 pixel coordinates)."""

    mask, origin, bin_um = compartment["mask"], compartment["origin"], compartment["bin_um"]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.contour(
        np.linspace(origin[0] / data.mpp, (origin[0] + mask.shape[1] * bin_um) / data.mpp, mask.shape[1]),
        np.linspace(origin[1] / data.mpp, (origin[1] + mask.shape[0] * bin_um) / data.mpp, mask.shape[0]),
        mask.astype(float),
        levels=[0.5],
        colors=color,
        linewidths=linewidth,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
