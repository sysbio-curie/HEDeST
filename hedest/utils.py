from __future__ import annotations

import io
import json
import os
import random
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
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
    norm: bool = False,
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
    print("Device found to load the model : ", device)

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
        print("spot_diameter_fullres already updated. Skipping.")
        return adata

    # Store old value
    if "spot_diameter_fullres" not in scalefactors:
        raise KeyError("spot_diameter_fullres not found in scalefactors.")

    scalefactors["spot_diameter0"] = scalefactors["spot_diameter_fullres"]

    # Update value (Visium spot diameter = 55 µm)
    scalefactors["spot_diameter_fullres"] = 55 / mpp

    return adata


def count_cell_types(seg_dict: Dict[str, Any], ct_list: List[str]) -> pd.DataFrame:
    """
    Counts cell types in the segmentation dictionary.

    Args:
        seg_dict: Dictionary containing segmentation data.
        ct_list: List of cell type names.

    Returns:
        DataFrame containing counts of each cell type.
    """

    cell_type_counts = {}
    nuc = seg_dict["nuc"]
    for cell_id in nuc.keys():
        label = nuc[cell_id]["type"]
        cell_type = ct_list[int(label)]
        if cell_type not in cell_type_counts.keys():
            cell_type_counts[cell_type] = 1
        else:
            cell_type_counts[cell_type] += 1
    df = pd.DataFrame([cell_type_counts])

    return df


def fig_to_array(fig: Figure) -> np.ndarray:
    """
    Converts a Matplotlib figure to a NumPy array.

    Args:
        fig: A Matplotlib figure.

    Returns:
        An image array.
    """

    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


def check_json_classification(data: Dict[str, Dict[str, Dict[str, Optional[str]]]]) -> bool:
    """
    Checks if all cells in the JSON classification data have a non-None type.

    Args:
        data: A nested dictionary containing classification data.

    Returns:
        True if all cells have types, False otherwise.
    """

    first_key = next(iter(data["nuc"]))
    return data["nuc"][first_key]["type"] is not None


def seg_colors_compatible(
    seg_dict: Dict[str, Dict[str, Dict[str, Union[str, int]]]], color_dict: Dict[str, Tuple[str, Tuple[int, int, int]]]
) -> bool:
    """
    Checks if segmentation labels are compatible with color dictionary.

    Args:
        seg_dict: Segmentation data dictionary.
        color_dict: Color dictionary.

    Returns:
        True if all segmentation labels have corresponding colors, False otherwise.
    """

    seg_labels = set(str(cell["type"]) for cell_data in seg_dict["nuc"].values() for cell in [cell_data])
    color_labels = set(color_dict.keys())

    return (seg_labels - color_labels) == set()


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


def remove_empty_keys(data: Dict[str, List]) -> Dict[str, List]:
    """
    Removes keys with empty lists from a dictionary.

    Args:
        data: Input dictionary.

    Returns:
        Dictionary without empty keys.
    """

    empty_keys = []
    for key, value in data.items():
        if value == []:
            empty_keys.append(key)

    for element in empty_keys:
        del data[element]

    return data


def require_attributes(*required_attributes: str) -> Callable:
    """
    Decorator to ensure required attributes of a class are not None.

    Args:
        *required_attributes: Names of required attributes.

    Returns:
        Callable: Decorated function.
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            missing_attrs = [attr for attr in required_attributes if getattr(self, attr, None) is None]
            if missing_attrs:
                raise ValueError(
                    f"Your object contains NoneType attribute(s): {', '.join(missing_attrs)}. "
                    "Please add them with the add_attributes function."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def generate_color_dict(
    labels: List[str], palette: str = "tab20", format: str = "classic", n_max: int = 40
) -> Dict[str, Union[Tuple, Tuple[str, List[int]]]]:
    """
    Generates a dictionary of colors for labels.

    Args:
        labels: List of class labels.
        palette: Matplotlib color palette.
        format: Output format - "classic" or "special".
        n_max: Maximum number of unique colors.

    Returns:
        Color dictionary.
    """

    if len(labels) > n_max:
        print("Warning: The number of classes is greater than the maximum number of colors available in the palette.")
        print("The colors will be repeated.")

    num_classes = len(labels)
    cmap = plt.get_cmap(palette)

    if format == "classic":
        return {labels[i]: cmap(i % n_max) for i in range(num_classes)}

    elif format == "special":
        color_dict = {}
        for i, class_name in enumerate(labels):
            color = cmap(i % n_max)
            color = [int(255 * c) for c in color]
            color_dict[str(i)] = [class_name, color]

        return color_dict

    else:
        raise ValueError("Format must be either 'classic' or 'special'.")


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
                             The "type" field is expected to be the class *name*
                             (string) when coming from PredAnalyzer.seg_dict_w_class,
                             or an integer index otherwise.
        geojson_output_path: Where to write the .geojson file.
        color_dict:          Optional dict in 'special' format:
                             { "0": ["ClassName", [R, G, B, A]], ... }
                             If None, all cells are colored white (-1).
    """

    # Build name -> colorRGB lookup
    color_lookup: Dict[str, int] = {}
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

    print(f"{len(features)} cells exported to {geojson_output_path}")
    if skipped:
        print(f"{skipped} cells skipped (contour < 3 points)")
