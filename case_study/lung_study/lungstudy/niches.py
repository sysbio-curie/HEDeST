"""
Cellular neighbourhoods and immune infiltration.

**Niches.** With the deconvolution, tissue regions are found by clustering the spots on
their proportions (one label per 55 µm spot). With HEDeST, each cell gets the composition
of its 30 nearest neighbours (a ~50 µm neighbourhood), and the cells are clustered on it
(cellular neighbourhoods, as in multiplexed imaging): one label per cell, wherever it is.
The two are fitted on all the slides together, with the same number of clusters.

**Infiltration.** The epithelial compartment is drawn from the HEDeST cells (20 µm smoothing,
epithelial cells at least half of the cells), and every cell gets its signed distance to its
border (negative inside). The density of each immune type is then measured from deep inside
the epithelium to far into the stroma. A spot only says how many T cells share it with
epithelial cells, not on which side of the border they are.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from lungstudy.data import SampleData
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as label_components
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

# ------------------------------------------------------------------ niches


def neighbourhood_composition(data: SampleData, types: Sequence[str], k: int = 30) -> pd.DataFrame:
    """Share of each type among the ``k`` nearest neighbours of every cell (itself excluded)."""

    xy = data.cells[["x_um", "y_um"]].to_numpy()
    _, neighbours = cKDTree(xy).query(xy, k=k + 1, workers=-1)
    codes = pd.Categorical(data.cells["label"], categories=list(types)).codes
    neighbour_codes = codes[neighbours[:, 1:]]
    counts = np.zeros((len(xy), len(types)))
    for j in range(len(types)):
        counts[:, j] = (neighbour_codes == j).sum(axis=1)

    return pd.DataFrame(counts / k, index=data.cells.index, columns=list(types))


def name_clusters(centroids: pd.DataFrame, min_share: float = 0.2) -> List[str]:
    """A readable name per cluster from its dominant types (e.g. 'B cell + T cell')."""

    names = []
    for _, row in centroids.iterrows():
        top = row.sort_values(ascending=False)
        parts = [t for t, v in top.items() if v >= min_share][:3] or [top.index[0]]
        names.append(" + ".join(parts))
    seen: Dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name} ({seen[name]})")

    return unique


def fit_niches(
    compositions: Dict[str, pd.DataFrame], n_clusters: int = 8, seed: int = 0, fit_size: int = 300_000
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """
    One KMeans on the pooled rows of several samples (cells or spots).

    Returns:
        The cluster name of every row, per sample, and the cluster centroids.
    """

    pooled = pd.concat(compositions.values())
    rng = np.random.default_rng(seed)
    fit_rows = pooled.to_numpy()
    if len(fit_rows) > fit_size:
        fit_rows = fit_rows[rng.choice(len(fit_rows), fit_size, replace=False)]
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(fit_rows)
    centroids = pd.DataFrame(model.cluster_centers_, columns=pooled.columns)
    # Order the clusters by their dominant type, so the same niches get the same colours.
    order = centroids.apply(lambda r: (list(pooled.columns).index(r.idxmax()), -r.max()), axis=1).sort_values().index
    names = name_clusters(centroids.loc[order])
    rename = dict(zip(order, names))
    centroids = centroids.loc[order]
    centroids.index = names
    labels = {
        sample: pd.Series(model.predict(frame.to_numpy()), index=frame.index).map(rename)
        for sample, frame in compositions.items()
    }

    return labels, centroids


def within_spot_diversity(
    data: SampleData, niches: pd.Series, min_cells: int = 8, min_share: float = 0.25
) -> pd.DataFrame:
    """Per spot: how many cellular niches share it (each holding at least ``min_share`` of its cells)."""

    inside = data.cells.loc[data.cells["in_spot"], ["spot"]].join(niches.rename("niche"))
    rows = []
    for spot, group in inside.groupby("spot"):
        if len(group) < min_cells:
            continue
        shares = group["niche"].value_counts(normalize=True)
        rows.append(
            {
                "spot": spot,
                "n_cells": len(group),
                "dominant_share": float(shares.iloc[0]),
                "n_niches": int((shares >= min_share).sum()),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ infiltration


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
) -> pd.DataFrame:
    """Density (cells per mm² of tissue) of each type by signed distance to the epithelial border."""

    signed_grid = compartment["signed"]
    tissue = compartment["tissue"]
    pixel_mm2 = (compartment["bin_um"] / 1000) ** 2
    distance = compartment["cells"]
    edges = np.asarray(edges_um, dtype=float)
    area = np.histogram(signed_grid[tissue], bins=edges)[0] * pixel_mm2
    rows = []
    for cell_type in types:
        counts = np.histogram(distance[data.cells["label"] == cell_type], bins=edges)[0]
        for i in range(len(edges) - 1):
            rows.append(
                {
                    "cell_type": cell_type,
                    "distance_um": (edges[i] + edges[i + 1]) / 2,
                    "density_per_mm2": counts[i] / area[i] if area[i] > 0 else np.nan,
                    "n": int(counts[i]),
                }
            )

    return pd.DataFrame(rows)
