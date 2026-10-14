from __future__ import annotations

from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from geopandas import GeoDataFrame
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from shapely.affinity import affine_transform
from shapely.geometry import Point
from shapely.validation import make_valid
from spatialdata.models import ShapesModel
from spatialdata.transformations import get_transformation
from tqdm import tqdm


def apply_transformation_to_gdf(gdf: GeoDataFrame, affine_matrix: tuple) -> GeoDataFrame:
    """
    Applies affine transformation to GeoDataFrame geometries.

    Args:
        gdf: Input GeoDataFrame with geometries to transform.
        affine_matrix: Affine transformation parameters in the form
                       (a, b, d, e, xoff, yoff).

    Returns:
        Transformed GeoDataFrame with updated geometries.
    """

    transformed_geometries = gdf.geometry.apply(lambda geom: affine_transform(geom, affine_matrix))
    return transformed_geometries


def match_nuclei_and_cells(gdf_nucleus: GeoDataFrame, gdf_cells: GeoDataFrame) -> GeoDataFrame:
    """
    Matches nuclei to cells based on spatial overlap.

    Args:
        gdf_nucleus: GeoDataFrame containing nucleus geometries.
        gdf_cells: GeoDataFrame containing cell geometries.

    Returns:
        GeoDataFrame with intersection areas and overlap ratios.
    """

    gdf_nucleus["geometry"] = gdf_nucleus["geometry"].apply(
        lambda geom: make_valid(geom) if not geom.is_valid else geom
    )
    gdf_cells["geometry"] = gdf_cells["geometry"].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

    intersections = gpd.sjoin(gdf_nucleus, gdf_cells, how="inner", predicate="intersects")
    intersections["index_left"] = intersections.index
    intersections["intersection_area"] = intersections.apply(
        lambda row: row["geometry"].intersection(gdf_cells.loc[row["index_right"], "geometry"]).area, axis=1
    )

    nucleus_areas = gdf_nucleus.copy()
    nucleus_areas["nucleus_area"] = nucleus_areas.geometry.area
    intersections = intersections.merge(nucleus_areas[["nucleus_area"]], left_index=True, right_index=True)

    intersections["overlap_ratio"] = intersections["intersection_area"] / intersections["nucleus_area"]

    return intersections


def filter_matches(intersections: GeoDataFrame, overlap_threshold: float = 0.95) -> GeoDataFrame:
    """
    Filters nucleus-cell matches based on overlap ratio.

    Args:
        intersections: GeoDataFrame with intersection areas and overlap ratios.
        overlap_threshold: Minimum overlap ratio to consider a valid match.

    Returns:
        GeoDataFrame with filtered nucleus-cell matches.
    """

    filtered_matches = intersections[intersections["overlap_ratio"] >= overlap_threshold]

    filtered_matches_sorted = filtered_matches.sort_values(by=["index_left", "overlap_ratio"], ascending=[True, False])
    nuclei_in_cells = filtered_matches_sorted.groupby("index_left", as_index=False).first()

    nuclei_in_cells = nuclei_in_cells[["index_left", "index_right", "overlap_ratio"]]
    nuclei_in_cells.columns = ["nucleus_id", "cell_id", "overlap_ratio"]

    nuclei_in_cells = nuclei_in_cells[~nuclei_in_cells["cell_id"].duplicated(keep=False)]
    nuclei_in_cells.reset_index(drop=True, inplace=True)

    return nuclei_in_cells


def make_spot_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, diameter: float, spacing: float
) -> GeoDataFrame:
    """
    Creates a hexagonal grid of circular spots within specified bounds.

    Args:
        x_min: Minimum x-coordinate of the bounding box.
        x_max: Maximum x-coordinate of the bounding box.
        y_min: Minimum y-coordinate of the bounding box.
        y_max: Maximum y-coordinate of the bounding box.
        diameter: Diameter of each circular spot.
        spacing: Distance between the centers of adjacent spots.

    Returns:
        GeoDataFrame containing the circular spots.
    """

    radius = diameter / 2
    circles = []

    y = y_min + radius
    row_index = 0

    while y + radius <= y_max:
        x_start = x_min + (radius if row_index % 2 == 0 else (spacing / 2) + radius)
        x = x_start

        while x + radius <= x_max:
            circle = Point(x, y).buffer(radius)
            circles.append(circle)
            x += spacing

        y += spacing
        row_index += 1

    grid_gdf = GeoDataFrame(circles, columns=["geometry"])
    grid_gdf.index = grid_gdf.index.astype(str)
    return ShapesModel.parse(grid_gdf)


def match_nuclei_in_spots(gdf_spots: GeoDataFrame, gdf_nucleus: GeoDataFrame) -> GeoDataFrame:
    """
    Matches nuclei to spots based on spatial containment.

    Args:
        gdf_spots: GeoDataFrame containing spot geometries.
        gdf_nucleus: GeoDataFrame containing nucleus geometries.

    Returns:
        GeoDataFrame with matched nucleus and spot IDs.
    """

    nucleus_centers = gdf_nucleus.copy()
    nucleus_centers["geometry"] = nucleus_centers.geometry.centroid

    affine_matrix = get_transformation(nucleus_centers).matrix

    shapely_affine = (
        affine_matrix[0, 0],  # a
        affine_matrix[0, 1],  # b
        affine_matrix[1, 0],  # d
        affine_matrix[1, 1],  # e
        affine_matrix[0, 2],  # xoff
        affine_matrix[1, 2],  # yoff
    )

    nucleus_centers_transformed = nucleus_centers.copy()
    nucleus_centers_transformed.geometry = apply_transformation_to_gdf(nucleus_centers_transformed, shapely_affine)

    nuclei_in_spots = gpd.sjoin(nucleus_centers_transformed, gdf_spots, how="inner", predicate="within")
    nuclei_in_spots["index_left"] = nuclei_in_spots.index

    nuclei_in_spots = nuclei_in_spots[["index_left", "index_right"]].rename(
        columns={"index_left": "nucleus_id", "index_right": "spot_id"}
    )
    nuclei_in_spots.reset_index(drop=True, inplace=True)

    return nuclei_in_spots


def match_nuclei_to_closest_spots(gdf_nucleus: GeoDataFrame, gdf_spots: GeoDataFrame) -> GeoDataFrame:
    """
    Matches each nucleus to the closest spot based on centroid distances.

    Args:
        gdf_nucleus: GeoDataFrame containing nucleus geometries.
        gdf_spots: GeoDataFrame containing spot geometries.

    Returns:
        DataFrame with matched nucleus and spot IDs along with distances.
    """

    if "geometry" not in gdf_nucleus.columns or "geometry" not in gdf_spots.columns:
        raise ValueError("Both GeoDataFrames must have a 'geometry' column with valid geometries.")

    nucleus_centers = gdf_nucleus.copy()
    nucleus_centers["geometry"] = nucleus_centers.geometry.centroid

    affine_matrix = get_transformation(nucleus_centers).matrix

    shapely_affine = (
        affine_matrix[0, 0],  # a
        affine_matrix[0, 1],  # b
        affine_matrix[1, 0],  # d
        affine_matrix[1, 1],  # e
        affine_matrix[0, 2],  # xoff
        affine_matrix[1, 2],  # yoff
    )

    nucleus_centers_transformed = nucleus_centers.copy()
    nucleus_centers_transformed.geometry = apply_transformation_to_gdf(nucleus_centers_transformed, shapely_affine)

    gdf_spots["centroid"] = gdf_spots.geometry.centroid

    nucleus_coords = np.array([[p.x, p.y] for p in nucleus_centers_transformed.geometry])
    spot_coords = np.array([[p.x, p.y] for p in gdf_spots["centroid"]])

    tree = cKDTree(spot_coords)
    distances, indices = tree.query(nucleus_coords)

    closest_spot_ids = gdf_spots.index[indices]

    result_df = pd.DataFrame({"nucleus_id": gdf_nucleus.index, "spot_id": closest_spot_ids, "distance": distances})

    return result_df


def add_cell_type_to_df(
    df: pd.DataFrame, adata: AnnData, cell_id_key: str = "cell_id", cell_type_key: str = "cell_type"
) -> pd.DataFrame:
    """
    Adds cell type annotations from an AnnData object to a DataFrame based on cell IDs.

    Args:
        df: DataFrame containing cell IDs.
        adata: AnnData object with cell type annotations in `obs`.
        cell_id_key: Column name in `df` corresponding to cell IDs.
        cell_type_key: Column name in `adata.obs` corresponding to cell type annotations.

    Returns:
        DataFrame with added cell type annotations.
    """

    table = adata.obs
    df = df.merge(table[["cell_id", cell_type_key]], on=cell_id_key, how="left")
    return df


def find_closest_nucleus(gdf_nucleus: GeoDataFrame, gdf_nucleus_hn: GeoDataFrame) -> pd.DataFrame:
    """
    Finds the closest nucleus in `gdf_nucleus` for each nucleus in `gdf_nucleus_hn`.

    Args:
        gdf_nucleus: GeoDataFrame containing nucleus geometries.
        gdf_nucleus_hn: GeoDataFrame containing high-nucleus geometries.

    Returns:
        DataFrame with closest nucleus IDs and distances.
    """

    if "geometry" not in gdf_nucleus.columns or "geometry" not in gdf_nucleus_hn.columns:
        raise ValueError("Both GeoDataFrames must have a 'geometry' column with valid geometries.")

    gdf_nucleus["centroid"] = gdf_nucleus.geometry.centroid
    gdf_nucleus_hn["centroid"] = gdf_nucleus_hn.geometry.centroid

    nucleus_coords = np.array([[p.x, p.y] for p in gdf_nucleus["centroid"]])
    hn_coords = np.array([[p.x, p.y] for p in gdf_nucleus_hn["centroid"]])

    tree = cKDTree(nucleus_coords)
    distances, indices = tree.query(hn_coords)

    closest_ids = gdf_nucleus.index[indices]

    result_df = pd.DataFrame({"nucleus_id": closest_ids, "distance": distances}, index=gdf_nucleus_hn.index)

    return result_df


def histogram_per_spot(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    color: str = "skyblue",
    title: Optional[str] = None,
    show_mean: bool = True,
) -> None:
    """
    Plots a histogram for the given dataframe. If no axis is provided, a new figure is created.

    Args:
        df: DataFrame containing `spot_id` column.
        ax: Matplotlib Axes object to plot on. If None, a new figure is created.
        color: Color of the histogram bars.
        title: Title of the plot.
        show_mean: Whether to display the mean line on the histogram.
    """

    nucleus_per_spot = df.groupby("spot_id").size()
    spots_per_nucleus_count = nucleus_per_spot.value_counts().sort_index()
    mean_nucleus_per_spot = nucleus_per_spot.mean()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.bar(spots_per_nucleus_count.index, spots_per_nucleus_count.values, color=color, edgecolor="black")
    ax.set_xlabel("Nucleus count per spot", fontsize=12)
    ax.set_ylabel("Number of spots", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if show_mean:
        ax.axvline(
            mean_nucleus_per_spot, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_nucleus_per_spot:.2f}"
        )
        ax.legend()

    if ax is None:
        plt.show()


def spot_gdf_to_df(gdf: GeoDataFrame) -> pd.DataFrame:
    """
    Transforms a GeoDataFrame of spots into a DataFrame with their center coordinates and diameter.

    Args:
        gdf: GeoDataFrame containing spot geometries.

    Returns:
        DataFrame with columns `x_center`, `y_center`, and `diameter`.
    """

    centers = gdf.geometry.centroid
    x_center = centers.x
    y_center = centers.y

    def calculate_diameter(geom):
        if geom.is_empty:
            return 0
        coords = geom.convex_hull.exterior.coords
        return max(
            np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) for i, (x1, y1) in enumerate(coords) for x2, y2 in coords[i + 1 :]
        )

    diameters = gdf.geometry.apply(calculate_diameter)

    df = pd.DataFrame({"x_center": x_center, "y_center": y_center, "diameter": diameters})

    return df


def transfer_annot_batched(
    sc_adata: AnnData, xenium_adata: AnnData, cell_type_key: str, min_counts: int, k_neighb: int, batch_size: int = 5000
) -> AnnData:
    """
    Computes the cell type annotation for xenium cells using batched processing to reduce memory usage.

    Args:
        sc_adata: The single-cell RNA-seq dataset.
        xenium_adata: The xenium dataset.
        cell_type_key: The key in `sc_adata.obs` corresponding to cell type annotations.
        k_neighb: Number of nearest neighbors to consider.
        batch_size: Number of xenium cells to process at a time.

    Returns:
        Annotated cell types for xenium cells.
        Annotation confidence scores.
    """

    rna_annot = sc_adata.obs[cell_type_key].values
    cm_genes = [g for g in xenium_adata.var_names if g in sc_adata.var_names]

    # Extract gene expression matrices
    rna_g_idx = [list(sc_adata.var_names).index(g) for g in cm_genes]
    rna_cm = np.array(sc_adata[:, rna_g_idx].X.todense())
    rna_mask = rna_cm.sum(1) > 0
    rna_cm = rna_cm[rna_mask]
    rna_annot = rna_annot[rna_mask]

    xenium_cm = np.array(xenium_adata[:, cm_genes].X.todense())
    xenium_cm_mask = xenium_cm.sum(1) > min_counts
    xenium_cm = xenium_cm[xenium_cm_mask]
    xenium_adata_filtered = xenium_adata[xenium_cm_mask].copy()

    n_xenium_cells = xenium_cm.shape[0]
    unique_cell_types = np.unique(np.append(rna_annot, "low_counts"))
    confidence_matrix = pd.DataFrame(0, index=xenium_adata_filtered.obs.index, columns=unique_cell_types)

    for i in tqdm(range(0, n_xenium_cells, batch_size), desc="Processing batches", unit="batch"):
        batch = xenium_cm[i : i + batch_size]
        batch_cost = cdist(batch, rna_cm, metric="cosine")
        nn = np.argsort(batch_cost, axis=1)[:, :k_neighb]
        nn_preds = rna_annot[nn]

        # Store confidence values for each cell
        for j, row in enumerate(nn_preds):
            cell_index = xenium_adata_filtered.obs.index[i + j]
            unique_vals, counts = np.unique(row, return_counts=True)
            proportions = counts / k_neighb
            confidence_matrix.loc[cell_index, unique_vals] = proportions

    # Handle unfiltered cells: Add "low_counts" rows for cells not in xenium_adata_filtered
    missing_cells = xenium_adata.obs.index.difference(xenium_adata_filtered.obs.index)
    missing_confidence = pd.DataFrame(0, index=missing_cells, columns=unique_cell_types)
    missing_confidence["low_counts"] = 1

    # Combine filtered and missing confidence matrices
    confidence_matrix = pd.concat([confidence_matrix, missing_confidence])
    confidence_matrix = confidence_matrix.reindex(xenium_adata.obs.index)

    xenium_adata.obsm["annotation_confidence"] = confidence_matrix

    return xenium_adata
