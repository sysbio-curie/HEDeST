"""
Spot geometry and cell-to-spot mapping.

A spatial transcriptomics spot is a disc on the slide. Everything HEDeST does with spots
needs two things: where the spots are, and which cells fall inside them.

**The diameter trap.** ``adata.uns['spatial'][name]['scalefactors']['spot_diameter_fullres']``
is written for visualization and is not reliable. The real diameter of a Visium spot is
55 µm, so in pixels it is ``55 / mpp``. Every function here takes an explicit ``mpp`` and
uses it when given, so a caller can never silently inherit the visualization value. Pass
``mpp`` whenever you know it.

**The cell-id convention.** Cells are identified by their position in the segmentation
file: the first nucleus of the JSON is ``"0"``, the second ``"1"``, and so on. HoVer-Net's
``run_infer.py`` re-indexes its output this way, so the ids here, the ones in the feature
dictionary and the ones in the prediction tables all refer to the same nuclei.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import numpy as np
from anndata import AnnData
from loguru import logger
from scipy.spatial import KDTree

VISIUM_SPOT_DIAMETER_UM = 55.0


class SpotGeometry(NamedTuple):
    """
    Where the spots are, in full-resolution slide pixels.

    Attributes:
        ids: Spot identifiers, in the order of the rows of ``centers``.
        centers: ``(n_spots, 2)`` array of spot centres, as (x, y).
        diameter: Spot diameter in pixels.
        from_mpp: True if the diameter was computed from an mpp, False if it was read from
            the AnnData scalefactors (in which case it is a visualization value).
    """

    ids: List[str]
    centers: np.ndarray
    diameter: float
    from_mpp: bool


def spot_geometry(adata: AnnData, adata_name: str, mpp: Optional[float] = None) -> SpotGeometry:
    """
    Reads spot centres and the spot diameter from an AnnData object.

    Args:
        adata: AnnData object holding the spatial transcriptomics data.
        adata_name: Key under ``adata.uns['spatial']``.
        mpp: Microns per pixel of the slide. When given, the diameter is
            ``55 / mpp``; otherwise it falls back to the scalefactors, which are
            only meant for visualization.

    Returns:
        The spot geometry.

    Raises:
        KeyError: If the spatial information cannot be found in the AnnData object.
    """

    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] not found: this does not look like spatial transcriptomics data.")

    centers = np.asarray(adata.obsm["spatial"], dtype="float64")

    if mpp is not None:
        if mpp <= 0:
            raise ValueError(f"mpp must be a positive float, got {mpp}.")
        return SpotGeometry(list(adata.obs.index), centers, VISIUM_SPOT_DIAMETER_UM / mpp, True)

    try:
        scalefactors = adata.uns["spatial"][adata_name]["scalefactors"]
        diameter = float(scalefactors["spot_diameter_fullres"])
    except (KeyError, TypeError) as err:
        raise KeyError(
            f"No spot diameter found for '{adata_name}' in adata.uns['spatial']. "
            "Either the sample name is wrong, or the object carries no scalefactors: "
            "pass mpp so the diameter can be computed as 55 / mpp instead."
        ) from err

    logger.warning(
        f"Using spot_diameter_fullres={diameter:.1f} px from the AnnData scalefactors, which is a "
        "visualization value. Pass mpp to use the real diameter (55 / mpp) instead."
    )
    return SpotGeometry(list(adata.obs.index), centers, diameter, False)


def load_seg_dict(seg: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns a segmentation dictionary, reading it from disk if needed.

    Args:
        seg: A segmentation dictionary, or the path to a HoVer-Net style JSON file.

    Returns:
        The segmentation dictionary, which holds the nuclei under the key ``"nuc"``.
    """

    if isinstance(seg, dict):
        data = seg
    elif isinstance(seg, str):
        if not seg.endswith(".json"):
            raise ValueError(f"A segmentation path must point to a .json file, got {seg}.")
        with open(seg) as json_file:
            data = json.load(json_file)
    else:
        raise TypeError(f"seg must be a dictionary or a path to a JSON file, got {type(seg).__name__}.")

    if "nuc" not in data:
        raise ValueError("The segmentation dictionary has no 'nuc' key.")

    return data


def cell_centroids(seg: Union[str, Dict[str, Any]]) -> tuple[List[str], np.ndarray]:
    """
    Extracts the centroid of every segmented nucleus.

    Args:
        seg: A segmentation dictionary, or the path to a HoVer-Net style JSON file.

    Returns:
        The cell ids and an ``(n_cells, 2)`` array of centroids, as (x, y), in the order
        of the segmentation file.
    """

    nuc = load_seg_dict(seg)["nuc"]
    ids = list(nuc.keys())
    centroids = np.asarray([nuc[cell_id]["centroid"] for cell_id in ids], dtype="float64")

    return ids, centroids


def map_cells_to_spots(
    adata: AnnData,
    adata_name: str,
    seg: Union[str, Dict[str, Any]],
    only_in: bool = True,
    mpp: Optional[float] = None,
) -> Dict[str, List[str]]:
    """
    Maps every spot to the cells it contains.

    Args:
        adata: AnnData object holding the spatial transcriptomics data.
        adata_name: Key under ``adata.uns['spatial']``.
        seg: A segmentation dictionary, or the path to a HoVer-Net style JSON file.
        only_in: If True (the default, and what HEDeST trains on), a cell belongs to a
            spot only when its centroid falls inside the disc, so a cell can be shared by
            two overlapping spots and cells between spots belong to none. If False, every
            cell is assigned to its closest spot, provided it lies within one radius of it.
        mpp: Microns per pixel of the slide. Strongly recommended: see the module docstring.

    Returns:
        A dictionary mapping each spot id to the list of its cell ids. Spots with no cell
        are left out.
    """

    geometry = spot_geometry(adata, adata_name, mpp=mpp)
    cell_ids, centroids = cell_centroids(seg)
    radius = geometry.diameter / 2.0

    tree = KDTree(geometry.centers)
    spot_cells: Dict[str, List[str]] = defaultdict(list)

    if only_in:
        # One query for every cell at once, then flatten; a cell inside two overlapping
        # spots is listed by both.
        matches = tree.query_ball_point(centroids, r=radius, workers=-1)
        for cell_index, spot_indices in enumerate(matches):
            for spot_index in spot_indices:
                spot_cells[geometry.ids[spot_index]].append(cell_ids[cell_index])
    else:
        distances, spot_indices = tree.query(centroids, workers=-1)
        for cell_index, (distance, spot_index) in enumerate(zip(distances, spot_indices)):
            if distance <= radius:
                spot_cells[geometry.ids[spot_index]].append(cell_ids[cell_index])

    n_mapped = sum(len(cells) for cells in spot_cells.values())
    logger.info(
        f"{len(spot_cells)}/{len(geometry.ids)} spots hold cells, {n_mapped} cell assignments "
        f"out of {len(cell_ids)} cells (only_in={only_in}, diameter={geometry.diameter:.1f} px)."
    )

    return dict(spot_cells)


def cells_in_spots(spot_dict: Dict[str, List[str]]) -> set[str]:
    """
    Returns the set of cells that belong to at least one spot.

    Args:
        spot_dict: A dictionary mapping spot ids to lists of cell ids.

    Returns:
        The set of cell ids.
    """

    return {cell_id for cell_ids in spot_dict.values() for cell_id in cell_ids}
