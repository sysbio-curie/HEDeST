"""
Which cells touch each other.

H&E shows nuclei, not cell borders, so a cell is drawn the way QuPath's cell expansion draws it:
its nucleus grown by up to 5 µm, never into another cell (every point within 5 µm of a nucleus
goes to the closest nucleus outline). Two cells **touch** when these shapes share a border.

How it is computed: the nucleus contours of the segmentation are drawn on a 0.5 µm raster, one
label per nucleus; ``skimage.segmentation.expand_labels`` grows them by 5 µm without overlap
(each pixel to the nearest nucleus, which is a Voronoi partition of the nucleus outlines); two
nuclei touch when their labels are side by side somewhere on the raster. The slide is processed in
1.5 mm tiles with a margin wide enough for any nucleus reaching into a tile.

Touching cells lie 5-16 µm apart (centre to centre; 5th-95th percentile), each cell touching 4
others (median): about the old "within 15 µm" neighbourhood, without its two weaknesses (a large
nucleus reached past its neighbours, and a cell "touched" another through a third one).
"""
from __future__ import annotations

from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd

GROW_UM = 5.0  # QuPath's default cell expansion
RESOLUTION_UM = 0.5
TILE_UM = 1500.0


def touching_pairs(
    cells: pd.DataFrame,
    contours: Dict[str, np.ndarray],
    mpp: float,
    grow_um: float = GROW_UM,
    resolution_um: float = RESOLUTION_UM,
    tile_um: float = TILE_UM,
) -> np.ndarray:
    """
    The pairs of touching cells.

    Args:
        cells: One row per nucleus, with ``x_um`` and ``y_um``; its order defines the indices.
        contours: Nucleus contour (level-0 pixels) of every cell id.
        mpp: Microns per level-0 pixel.

    Returns:
        An (n_pairs, 2) int32 array of row positions in ``cells``, each pair once (i < j).
    """

    import cv2
    from skimage.segmentation import expand_labels

    xy = cells[["x_um", "y_um"]].to_numpy()
    ids = list(cells.index)
    scale = mpp / resolution_um
    margin = 4 * grow_um + 20.0
    grow_px = int(round(grow_um / resolution_um))
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
    found = []
    for tx in np.arange(x0, x1 + 1, tile_um):
        for ty in np.arange(y0, y1 + 1, tile_um):
            core = (xy[:, 0] >= tx) & (xy[:, 0] < tx + tile_um) & (xy[:, 1] >= ty) & (xy[:, 1] < ty + tile_um)
            if not core.any():
                continue
            near = np.flatnonzero(
                (xy[:, 0] >= tx - margin)
                & (xy[:, 0] < tx + tile_um + margin)
                & (xy[:, 1] >= ty - margin)
                & (xy[:, 1] < ty + tile_um + margin)
            )
            offset = np.array([(tx - margin) / resolution_um, (ty - margin) / resolution_um])
            size = int(np.ceil((tile_um + 2 * margin) / resolution_um)) + 2
            raster = np.zeros((size, size), dtype=np.int32)
            for k in near:
                outline = np.round(contours[ids[k]].astype(np.float64) * scale - offset).astype(np.int32)
                cv2.fillPoly(raster, [outline.reshape(-1, 1, 2)], int(k) + 1)
            raster = expand_labels(raster, grow_px)
            for left, right in ((raster[:, :-1], raster[:, 1:]), (raster[:-1, :], raster[1:, :])):
                border = (left != right) & (left > 0) & (right > 0)
                found.append(np.stack([left[border], right[border]], axis=1) - 1)
    pairs = np.sort(np.concatenate(found), axis=1)

    return np.unique(pairs, axis=0).astype(np.int32)


def neighbour_counts(pairs: np.ndarray, codes: np.ndarray, n_types: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per cell: how many cells it touches, and how many of each type.

    Args:
        pairs: Touching pairs (row positions).
        codes: Type code of every cell (0..n_types-1, or -1 for a type not counted).

    Returns:
        ``degree`` (n_cells,) and ``counts`` (n_types, n_cells).
    """

    n = len(codes)
    degree = np.bincount(pairs.ravel(), minlength=n)
    counts = np.zeros((n_types, n))
    for j in range(n_types):
        to_j = codes[pairs[:, 1]] == j
        from_j = codes[pairs[:, 0]] == j
        counts[j] = np.bincount(pairs[to_j, 0], minlength=n) + np.bincount(pairs[from_j, 1], minlength=n)

    return degree, counts
