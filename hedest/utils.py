from __future__ import annotations

import json
import os
import random
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from loguru import logger
from shapely.geometry import Polygon
from shapely.validation import make_valid

from hedest.model.cell_classifier import CellClassifier


def set_seed(seed: int) -> None:
    """
    Sets the seed for random number generators in Python libraries.

    Args:
        seed: The seed value to set for random number generators.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(
    model_path: str,
    num_classes: int,
    embed_size: int,
    hidden_dims: List[int],
    norm: bool = True,
    dropout: float = 0.0,
) -> CellClassifier:
    """
    Loads a trained model from a file.

    Args:
        model_path: Path to the model file.
        num_classes: Number of classes in the model.
        embed_size: Size of the cell embeddings the model was trained on.
        hidden_dims: List of hidden layer dimensions.
        norm: Whether the model uses LayerNorm.
        dropout: Dropout rate used in the model.

    Returns:
        The loaded model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading the model on {device}")

    model = CellClassifier(
        num_classes=num_classes,
        embed_size=embed_size,
        hidden_dims=hidden_dims,
        norm=norm,
        dropout=dropout,
        device=device,
    )

    model.load_state_dict(torch.load(model_path, map_location=device))

    if device == torch.device("cuda"):
        model = model.to(device)

    return model


def load_spatial_adata(path: str):
    """
    Loads spatial transcriptomics data with Scanpy.

    Args:
        path: Path to an `.h5ad` file **or** a Visium directory.

    Returns:
        adata: The loaded AnnData object.
    """

    try:
        if os.path.isfile(path):
            return sc.read_h5ad(path)
        else:
            raise FileNotFoundError(f"'{path}' is not a valid file path.")
    except Exception as h5ad_error:
        try:
            return sc.read_visium(path)
        except Exception as visium_error:
            raise RuntimeError("Failed to load data with either sc.read_h5ad or sc.read_visium") from (
                h5ad_error or visium_error
            )


def update_spot_diameter(adata: AnnData, adata_name: str, mpp: float) -> AnnData:
    """
    Update spot_diameter_fullres in an AnnData object using a given microns-per-pixel (mpp).

    If 'spot_diameter0' already exists, the function assumes the update
    was already performed and does nothing.

    Args:
        adata: AnnData object containing spatial transcriptomics data.
        adata_name: Name of the sample in adata.uns['spatial'].
        mpp: Microns per pixel of the WSI.

    Returns:
        Updated AnnData object with modified spot diameter.
    """

    if mpp is None or mpp <= 0:
        raise ValueError("mpp must be a positive float.")

    scalefactors = adata.uns["spatial"][adata_name]["scalefactors"]

    # If already updated, skip
    if "spot_diameter0" in scalefactors:
        logger.info("spot_diameter_fullres was already updated, skipping.")
        return adata

    # Store old value
    if "spot_diameter_fullres" not in scalefactors:
        raise KeyError("spot_diameter_fullres not found in scalefactors.")

    scalefactors["spot_diameter0"] = scalefactors["spot_diameter_fullres"]

    # Update value (Visium spot diameter = 55 µm)
    scalefactors["spot_diameter_fullres"] = 55 / mpp

    return adata


def format_time(seconds: int) -> str:
    """
    Formats time duration in HH:MM:SS or MM:SS format.

    Args:
        seconds: Time duration in seconds.

    Returns:
        Formatted time string.
    """

    formatted_time = str(timedelta(seconds=int(seconds)))
    if seconds < 3600:
        formatted_time = formatted_time[2:]
    return formatted_time


def revert_dict(data: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Reverts a dictionary with lists as values to a dictionary mapping values to keys.

    Args:
        data: Input dictionary.

    Returns:
        Reverted dictionary.
    """

    return {val: key for key, values in data.items() for val in values}


def rgba_to_colorRGB(rgba: Tuple) -> int:
    """Convert an RGBA tuple (0-255 ints) to a signed 32-bit colorRGB integer (QuPath format)."""

    r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
    value = (r << 16) | (g << 8) | b
    # Convert to signed 32-bit
    if value >= 0x800000:
        value -= 0x1000000
    return value


def build_color_lookup(color_dict: Dict) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Convert a color_dict in 'special' format:
        { "0": ["TypeName", [R, G, B, A]], ... }
    into:
        - color_lookup:  { 0: colorRGB_int, 1: colorRGB_int, ... }
        - name_lookup:   { 0: "TypeName", 1: "TypeName", ... }
    """
    color_lookup = {}
    name_lookup = {}
    for key, (class_name, rgb) in color_dict.items():
        idx = int(key)
        color_lookup[idx] = rgba_to_colorRGB(rgb)
        name_lookup[idx] = class_name
    return color_lookup, name_lookup


def seg_dict_to_geojson(
    seg_dict: Dict,
    geojson_output_path: str,
    color_dict: Optional[Dict] = None,
) -> None:
    """
    Convert a HEDeST-annotated seg_dict (same format as HoVerNet JSON, with
    cell type labels assigned by HEDeST) into a QuPath-compatible GeoJSON file.

    Args:
        seg_dict:            Dictionary in HoVerNet nuc format, i.e.:
                             { "nuc": { cell_id: { "contour": [...],
                                                    "type": int_or_str,
                                                    "type_prob": float,
                                                    ... } } }
                             The "type" field holds the class index, as written by
                             PredAnalyzer.seg_dict_with_labels().
        geojson_output_path: Where to write the .geojson file.
        color_dict:          Optional dict in 'special' format:
                             { "0": ["ClassName", [R, G, B, A]], ... }
                             If None, all cells are colored white (-1).
    """

    # Build class index -> colorRGB lookup
    color_lookup: Dict[int, int] = {}
    name_lookup: Dict[int, str] = {}

    if color_dict is not None:
        color_lookup, name_lookup = build_color_lookup(color_dict)

    features = []
    skipped = 0

    for cell_id, cell_info in seg_dict["nuc"].items():
        contour = cell_info.get("contour", [])
        if len(contour) < 3:
            skipped += 1
            continue

        coords = [[float(p[0]), float(p[1])] for p in contour]
        poly = Polygon(coords)

        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        elif poly.geom_type != "Polygon":
            poly = poly.convex_hull

        cell_type = cell_info.get("type", 0)
        cell_type_int = int(cell_type)
        cell_type_name = name_lookup.get(cell_type_int, f"Type_{cell_type_int}")
        color_rgb = color_lookup.get(cell_type_int, -1)

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(poly.exterior.coords)],
                },
                "properties": {
                    "object_type": "detection",
                    "classification": {
                        "name": cell_type_name,  # human-readable name for QuPath
                        "colorRGB": color_rgb,
                    },
                    "isLocked": False,
                    "cell_id": str(cell_id),
                    "cell_type": cell_type_int,
                    "type_prob": cell_info.get("type_prob", None),
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    with open(geojson_output_path, "w") as f:
        json.dump(geojson, f)

    logger.info(f"{len(features)} cells exported to {geojson_output_path}")
    if skipped:
        logger.info(f"{skipped} cells skipped (contour < 3 points)")
