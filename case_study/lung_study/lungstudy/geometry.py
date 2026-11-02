"""
Distances to outlines on a raster of the slide, and cell densities by distance.

Used for "how dense is type X at d µm from the border of Y" questions: plasma cells around the
lymphoid aggregates, immune cells around the epithelium. The slide is cut in ``bin_um`` pixels;
a pixel is tissue when the smoothed nucleus density is high enough. Densities are cells per mm²
of *tissue* at each distance, so they do not depend on how much tissue lies at that distance.

The same questions at the spots' resolution (``spot_rings``): HEDeST only says which spots lie in
the structure (their centre inside it), and every spot is placed by how many rings of spots
separate it from them, spot to spot.
"""
from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lungstudy.data import SampleData
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.draw import polygon as fill_polygon


@dataclass
class Grid:
    """A raster of the slide in µm: pixel (row, col) covers origin + (col, row) * bin_um."""

    origin: np.ndarray
    bin_um: float
    shape: tuple
    cell_rows: np.ndarray
    cell_cols: np.ndarray
    tissue: np.ndarray

    def pixel_mm2(self) -> float:
        return (self.bin_um / 1000) ** 2

    def at(self, xy_um: np.ndarray) -> tuple:
        """The (rows, cols) of points given in µm, clipped to the grid."""

        cols = np.clip(((xy_um[:, 0] - self.origin[0]) / self.bin_um).astype(int), 0, self.shape[1] - 1)
        rows = np.clip(((xy_um[:, 1] - self.origin[1]) / self.bin_um).astype(int), 0, self.shape[0] - 1)
        return rows, cols


def tissue_grid(data: SampleData, bin_um: float = 5.0, sigma_um: float = 20.0) -> Grid:
    """The grid of a slide, with its tissue mask (smoothed nucleus density above a quarter of its median)."""

    xy = data.cells[["x_um", "y_um"]].to_numpy()
    origin = xy.min(axis=0) - 10 * bin_um
    shape = tuple((np.ceil((xy.max(axis=0) - origin) / bin_um) + 10).astype(int))[::-1]
    cols = ((xy[:, 0] - origin[0]) / bin_um).astype(int)
    rows = ((xy[:, 1] - origin[1]) / bin_um).astype(int)
    total = np.zeros(shape)
    np.add.at(total, (rows, cols), 1)
    smooth = gaussian_filter(total, sigma_um / bin_um)
    tissue = smooth > 0.25 * np.median(smooth[total > 0])

    return Grid(origin, bin_um, shape, rows, cols, tissue)


def rasterize(polygons: Iterable, grid: Grid) -> np.ndarray:
    """The pixels inside any of the polygons (µm coordinates; holes ignored)."""

    mask = np.zeros(grid.shape, dtype=bool)
    for polygon in polygons:
        parts = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
        for part in parts:
            coordinates = np.asarray(part.exterior.coords)
            rows, cols = fill_polygon(
                (coordinates[:, 1] - grid.origin[1]) / grid.bin_um,
                (coordinates[:, 0] - grid.origin[0]) / grid.bin_um,
                grid.shape,
            )
            mask[rows, cols] = True

    return mask


def signed_distance(mask: np.ndarray, bin_um: float) -> np.ndarray:
    """Distance of every pixel to the border of ``mask`` (µm): negative inside, positive outside."""

    return np.where(mask, -distance_transform_edt(mask) * bin_um, distance_transform_edt(~mask) * bin_um)


def density_profile(
    grid: Grid,
    distance: np.ndarray,
    labels: pd.Series,
    types: Sequence[str],
    edges_um: Sequence[float],
) -> pd.DataFrame:
    """
    Cells per mm² of tissue of each type, and its share of all the cells, by distance bin.

    Args:
        distance: The signed distance raster (µm).
        labels: The type of every cell of the grid's slide, in the order of ``grid.cell_rows``.
    """

    edges = np.asarray(edges_um, dtype=float)
    area = np.histogram(distance[grid.tissue], bins=edges)[0] * grid.pixel_mm2()
    cell_distance = distance[grid.cell_rows, grid.cell_cols]
    names = np.asarray(labels)
    total = np.histogram(cell_distance, bins=edges)[0]
    rows = []
    for cell_type in types:
        counts = np.histogram(cell_distance[names == cell_type], bins=edges)[0]
        for i in range(len(edges) - 1):
            rows.append(
                {
                    "cell_type": cell_type,
                    "distance_um": (edges[i] + edges[i + 1]) / 2,
                    "low_um": edges[i],
                    "high_um": edges[i + 1],
                    "density_per_mm2": counts[i] / area[i] if area[i] > 0 else np.nan,
                    "share": counts[i] / total[i] if total[i] > 0 else np.nan,
                    "n": int(counts[i]),
                }
            )

    return pd.DataFrame(rows)


def spot_rings(data: SampleData, inside: pd.Series, signed: bool = False, max_ring: int = 6) -> pd.Series:
    """
    Distance of every spot to a structure, in rings of spots on the hexagonal grid (one ring ≈ one
    pitch, 100 µm centre to centre).

    Args:
        inside: True for the spots whose centre lies in the structure (index: spot ids).
        signed: False: the inside spots are 0. True: they are minus the rings to the nearest
            outside spot (-1: inside, touching the outside; -2: one ring deeper, ...).
        max_ring: Rings beyond are clipped to it (shown as "≥").

    Returns:
        Outside spots: the rings to the nearest inside spot (1: touching one, 2: the next ring, ...).
    """

    xy = data.spots.loc[inside.index, ["x_um", "y_um"]].to_numpy()
    flags = inside.to_numpy(dtype=bool)
    pitch = data.pitch_px * data.mpp
    ring = np.zeros(len(flags), dtype=int)
    if flags.any() and (~flags).any():
        # hexagonal grid: ring k lies between 0.87k and k pitches away
        ring[~flags] = np.ceil(cKDTree(xy[flags]).query(xy[~flags])[0] / pitch - 0.05)
        if signed:
            ring[flags] = -np.ceil(cKDTree(xy[~flags]).query(xy[flags])[0] / pitch - 0.05)

    return pd.Series(np.clip(ring, -max_ring, max_ring), index=inside.index, name="ring")
