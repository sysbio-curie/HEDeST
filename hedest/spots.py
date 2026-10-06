"""
Spot geometry, cell-to-spot mapping, and the spots drawn as pies for QuPath.

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

**The spot pies.** ``export_spot_pies`` draws every spot as a pie of its cell-type
proportions, in a GeoJSON to load in QuPath together with the cell GeoJSON of a run. It needs
no trained model, only the proportions and the spot positions.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import yaml
from anndata import AnnData
from loguru import logger
from scipy.spatial import KDTree

VISIUM_SPOT_DIAMETER_UM = 55.0
SPOT_PIES_NAME = "spot_pies.geojson"
PIE_POINTS = 64  # points on the outline of a whole pie


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


def _wedge(x: float, y: float, radius: float, start: float, end: float) -> List[List[float]]:
    """
    Outline of a pie wedge, in slide pixels.

    Args:
        x, y: Centre of the pie.
        radius: Its radius.
        start, end: Where the wedge starts and ends, as fractions of a turn, clockwise from
            12 o'clock (y points down in the slide).

    Returns:
        The closed outline, as [x, y] points.
    """

    steps = max(1, math.ceil((end - start) * PIE_POINTS))
    arc = [
        [
            round(x + radius * math.sin(2 * math.pi * angle), 1),
            round(y - radius * math.cos(2 * math.pi * angle), 1),
        ]
        for angle in (start + (end - start) * i / steps for i in range(steps + 1))
    ]
    if end - start > 1 - 1e-6:  # the whole pie: a disc, already closed
        return arc
    centre = [round(x, 1), round(y, 1)]

    return [centre, *arc, centre]


def export_spot_pies(
    proportions: Union[str, Path, pd.DataFrame],
    adata: Union[str, Path, AnnData],
    path: Union[str, Path],
    mpp: Optional[float] = None,
    adata_name: Optional[str] = None,
    palette: Any = None,
) -> int:
    """
    Writes the cell-type proportions of the spots as pies, in a GeoJSON for QuPath.

    Every spot is drawn at its centre with its diameter, one wedge per cell type present,
    whose angle is the proportion of the cell type. The proportions are normalised per spot,
    as HEDeST trains on them, and the wedges start at 12 o'clock and follow the columns of the
    table, so a cell type always sits at the same place. A spot missing from the AnnData, or
    whose row holds no positive proportion, is not drawn.

    Loaded in QuPath together with the cell GeoJSON of a run, the pies show the two scales on
    top of each other with the same colours: the cells are *detections* and the wedges
    *annotations*, so 'D' shows or hides the cells, 'A' the pies, 'F' and 'Shift+F' fill them.
    Every wedge is locked and carries its proportion as a measurement, and the spot barcode as
    metadata (QuPath >= 0.5).

    Args:
        proportions: The proportions (spots x cell types), or the path of their CSV.
        adata: The AnnData object of the slide, or its path, for the spot centres.
        path: The GeoJSON to write.
        mpp: Microns per pixel of the slide, so the spots get their real diameter (55 / mpp).
            Strongly recommended: see the module docstring.
        adata_name: Key under ``adata.uns['spatial']``, only read without mpp. Defaults to
            the first one.
        palette: The colours: a ``Palette``, or the path of a YAML colour dictionary (special
            format). Defaults to the palette main.py and aggregate_seeds.py build from the
            cell types, so the pies take the colours of the cell GeoJSON of the run.

    Returns:
        The number of spots drawn.
    """

    from hedest.analysis.palette import Palette
    from hedest.analysis.palette import palette_from_yaml
    from hedest.dataset_utils import pp_prop
    from hedest.utils import load_spatial_adata
    from hedest.utils import rgba_to_colorRGB

    table = pp_prop(proportions.copy() if isinstance(proportions, pd.DataFrame) else str(proportions))

    if not isinstance(adata, AnnData):
        adata = load_spatial_adata(str(adata))
    if mpp is None and adata_name is None:
        adata_name = next(iter(adata.uns.get("spatial", {})), None)
    geometry = spot_geometry(adata, adata_name, mpp=mpp)
    centres = dict(zip(map(str, geometry.ids), geometry.centers[:, :2]))
    radius = geometry.diameter / 2

    missing = [spot for spot in table.index if spot not in centres]
    if missing:
        logger.warning(f"{len(missing)} spot(s) of the proportions are not in the AnnData and are not drawn.")

    if palette is None:
        palette = Palette(list(table.columns))
    elif not isinstance(palette, Palette):
        with open(palette) as color_file:
            palette = palette_from_yaml(yaml.safe_load(color_file))
    classes = {name: {"name": name, "colorRGB": rgba_to_colorRGB(palette.rgb255(name))} for name in table.columns}

    features = []
    drawn = 0
    for spot, row in table.iterrows():
        if spot not in centres:
            continue
        x, y = centres[spot]
        start = 0.0
        for name, value in row.items():
            if not value > 0:  # also skips NaN
                continue
            end = min(1.0, start + value)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [_wedge(x, y, radius, start, end)]},
                    "properties": {
                        "objectType": "annotation",
                        "classification": classes[name],
                        "isLocked": True,
                        "measurements": [{"name": "Proportion", "value": round(float(value), 4)}],
                        "metadata": {"spot": spot},
                    },
                }
            )
            start = end
        drawn += start > 0

    # Written aside then moved, so an interrupted export never leaves a truncated file behind
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    os.replace(tmp, path)

    logger.info(f"{drawn} spots drawn as pies ({len(features)} wedges) in {path}")

    return drawn
