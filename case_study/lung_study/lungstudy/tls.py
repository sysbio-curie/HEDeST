"""
Lymphoid aggregates (candidate tertiary lymphoid structures), from cells and from spots.

From the HEDeST cells (``detect_cell_aggregates``):

1. a nucleus is lymphoid when P(B cell) + P(T cell) > 0.5 at lv3 (mean of the 5 seeds,
   without PPSA). Summing the two probabilities keeps the detection robust to HEDeST
   hesitating between B and T, the hard distinction on H&E; NK cells are left out because
   their lv3 proportions follow the epithelium in these slides;
2. positions in µm;
3. DBSCAN on the lymphoid nuclei: eps = 15 µm, min_samples = 10 (at least 10 lymphoid
   nuclei within 15 µm, i.e. one per ~70 µm²: dense packing). Planned at 20 µm; on
   A_AAAA1879, 20 µm chained the peritumoral infiltrates into rings around the tumour nests,
   so it was tightened to 15 µm before any comparison (the other values are reported as a
   sensitivity analysis);
4. a cluster is kept with at least 50 lymphoid nuclei and a purity of at least 60% (lymphoid
   nuclei among all nuclei inside its outline);
5. outline: concave hull of its nuclei, grown by 5 µm (a nucleus radius);
6. described by its size, its B:T balance, how B and T cells are arranged inside it (mixed or
   in separate zones), its B-cell follicles, and its lv4 composition (plasma cells, CD4 / CD8).

From the spots (``detect_spot_regions``), what a deconvolution allows: spots whose B + T
proportion reaches a threshold, grouped with their neighbours into regions; a region's
smallest unit is one spot, and it has no inside.

A reference that owes nothing to the transcriptomics nor to HEDeST: the same DBSCAN on the
*small round nuclei* of the segmentation (area < 25 µm², circularity > 0.8), the shape of
lymphocytes. HoVer-Net's own *inflammatory* class cannot serve: on these slides it calls
70-90% of all nuclei, lymphocytes included, *neoplastic*.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import shapely
from lungstudy.data import SampleData
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

LYMPHOID = ["B cell", "T cell"]
DEFAULTS = dict(threshold=0.5, eps_um=15.0, min_samples=10, min_cells=50, min_purity=0.6)
SMALL_ROUND = dict(max_area_um2=25.0, min_circularity=0.8)


def lymphoid_score(data: SampleData, types: Sequence[str] = LYMPHOID) -> pd.Series:
    return data.pred[list(types)].sum(axis=1).reindex(data.cells.index)


def small_round_score(data: SampleData) -> pd.Series:
    """1 for the nuclei shaped like lymphocytes (small and round), from the segmentation alone."""

    cells = data.cells
    small = (cells["area_um2"] < SMALL_ROUND["max_area_um2"]) & (cells["circularity"] > SMALL_ROUND["min_circularity"])

    return small.astype(float)


def detect_cell_aggregates(
    data: SampleData,
    score: pd.Series,
    threshold: float = 0.5,
    eps_um: float = 15.0,
    min_samples: int = 10,
    min_cells: int = 50,
    min_purity: float = 0.6,
    hull_ratio: float = 0.25,
    grow_um: float = 5.0,
) -> pd.DataFrame:
    """
    Dense, pure clusters of the nuclei whose ``score`` exceeds ``threshold``.

    Returns:
        One row per aggregate: polygon (µm), area, nuclei counts, purity, centroid (px), and
        the ids of all the nuclei inside it.
    """

    cells = data.cells
    positive = (score > threshold).to_numpy()
    xy = cells[["x_um", "y_um"]].to_numpy()
    if positive.sum() < min_samples:
        return _empty()

    clusters = DBSCAN(eps=eps_um, min_samples=min_samples, n_jobs=-1).fit_predict(xy[positive])
    positive_index = np.flatnonzero(positive)
    tree = cKDTree(xy)
    rows = []
    for cluster in np.unique(clusters[clusters >= 0]):
        members = positive_index[clusters == cluster]
        if len(members) < min_cells:
            continue
        polygon = shapely.concave_hull(MultiPoint(xy[members]), ratio=hull_ratio).buffer(grow_um)
        x0, y0, x1, y1 = polygon.bounds
        candidates = tree.query_ball_point([(x0 + x1) / 2, (y0 + y1) / 2], r=np.hypot(x1 - x0, y1 - y0) / 2 + 1)
        candidates = np.asarray(candidates, dtype=int)
        inside = candidates[shapely.contains_xy(polygon, xy[candidates, 0], xy[candidates, 1])]
        purity = float(positive[inside].mean()) if len(inside) else 0.0
        if purity < min_purity:
            continue
        rows.append(
            {
                "polygon_um": polygon,
                "area_um2": float(polygon.area),
                "n_cells": int(len(inside)),
                "n_positive": int(positive[inside].sum()),
                "purity": purity,
                "x": float(polygon.centroid.x / data.mpp),
                "y": float(polygon.centroid.y / data.mpp),
                "cell_ids": list(cells.index[inside]),
                "share_in_spots": float(cells["in_spot"].to_numpy()[inside].mean()),
            }
        )
    if not rows:
        return _empty()
    table = pd.DataFrame(rows).sort_values("area_um2", ascending=False).reset_index(drop=True)
    table.index = [f"{data.name}_A{i + 1:03d}" for i in range(len(table))]

    return table


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["polygon_um", "area_um2", "n_cells", "n_positive", "purity", "x", "y", "cell_ids", "share_in_spots"]
    )


def detect_spot_regions(
    data: SampleData, types: Sequence[str] = LYMPHOID, threshold: float = 0.5, min_spots: int = 1
) -> pd.DataFrame:
    """
    Regions of neighbouring spots whose summed proportion of ``types`` reaches ``threshold``.

    Returns:
        One row per region: its spots, polygon (µm, union of the spots' hexagonal cells),
        area (number of spots x area of a grid cell), centroid (px).
    """

    score = data.props[list(types)].sum(axis=1)
    selected = score.index[score >= threshold]
    if not len(selected):
        return pd.DataFrame(columns=["spots", "n_spots", "polygon_um", "area_um2", "x", "y", "mean_score"])
    xy = data.spots.loc[selected, ["x", "y"]].to_numpy()
    graph = cKDTree(xy).sparse_distance_matrix(cKDTree(xy), 1.2 * data.pitch_px)
    _, labels = connected_components(graph, directed=False)
    pitch_um = data.pitch_px * data.mpp
    cell_area = np.sqrt(3) / 2 * pitch_um**2
    hex_radius = pitch_um / np.sqrt(3) * 1.02
    rows = []
    for label in np.unique(labels):
        members = selected[labels == label]
        if len(members) < min_spots:
            continue
        centres = data.spots.loc[members, ["x_um", "y_um"]].to_numpy()
        polygon = shapely.unary_union([Point(x, y).buffer(hex_radius, quad_segs=2) for x, y in centres])
        rows.append(
            {
                "spots": list(members),
                "n_spots": len(members),
                "polygon_um": polygon,
                "area_um2": float(len(members) * cell_area),
                "x": float(centres[:, 0].mean() / data.mpp),
                "y": float(centres[:, 1].mean() / data.mpp),
                "mean_score": float(score.loc[members].mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("area_um2", ascending=False).reset_index(drop=True)
    table.index = [f"{data.name}_S{i + 1:03d}" for i in range(len(table))]

    return table


# ------------------------------------------------------------------ comparison


def overlaps(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Every pair of overlapping polygons between two tables, with the intersection over union."""

    rows = []
    if not len(left) or not len(right):
        return pd.DataFrame(columns=["left", "right", "intersection_um2", "iou", "share_of_left", "share_of_smaller"])
    right_tree = shapely.STRtree(list(right["polygon_um"]))
    right_ids = list(right.index)
    for left_id, polygon in left["polygon_um"].items():
        for j in right_tree.query(polygon, predicate="intersects"):
            other = right["polygon_um"].iloc[j]
            inter = polygon.intersection(other).area
            union = polygon.union(other).area
            rows.append(
                {
                    "left": left_id,
                    "right": right_ids[j],
                    "intersection_um2": inter,
                    "iou": inter / union if union else 0.0,
                    "share_of_left": inter / polygon.area if polygon.area else 0.0,
                    "share_of_smaller": inter / min(polygon.area, other.area) if min(polygon.area, other.area) else 0.0,
                }
            )

    return pd.DataFrame(rows)


def detection_scores(found: pd.DataFrame, reference: pd.DataFrame, min_share: float = 0.2) -> Dict[str, float]:
    """
    Recall (share of the reference aggregates matched by a detection), precision (share of
    the detections matching a reference aggregate) and F1. Two polygons match when their
    intersection covers at least ``min_share`` of the smaller one, so that a coarse region
    holding a small aggregate counts as finding it.
    """

    if not len(reference) or not len(found):
        return {
            "recall": np.nan if not len(reference) else 0.0,
            "precision": np.nan if not len(found) else 0.0,
            "f1": np.nan,
        }
    pairs = overlaps(reference, found)
    matched = pairs[pairs["share_of_smaller"] >= min_share] if len(pairs) else pairs
    covered = set(matched["left"]) if len(matched) else set()
    real = set(matched["right"]) if len(matched) else set()
    recall = len(covered) / len(reference)
    precision = len(real) / len(found)
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0

    return {"recall": recall, "precision": precision, "f1": f1}


# ------------------------------------------------------------------ description


def describe_aggregates(
    aggregates: pd.DataFrame, data: SampleData, lv4: Optional[SampleData] = None, ring_um: float = 30.0
) -> pd.DataFrame:
    """
    What is inside each HEDeST aggregate: B:T balance, B share from the rim to the core, and
    with lv4 the plasma cells and the CD4 / CD8 T cells, inside and in a ring around it.
    """

    labels = data.cells["label"]
    xy = data.cells[["x_um", "y_um"]].to_numpy()
    index_of = pd.Series(np.arange(len(data.cells)), index=data.cells.index)
    tree = cKDTree(xy)
    rows = []
    for aggregate_id, row in aggregates.iterrows():
        polygon = row["polygon_um"]
        members = index_of.loc[row["cell_ids"]].to_numpy()
        names = labels.to_numpy()[members]
        n_b, n_t = int((names == "B cell").sum()), int((names == "T cell").sum())
        depth = np.array([polygon.exterior.distance(Point(p)) for p in xy[members]])
        relative = depth / depth.max() if depth.max() > 0 else depth
        lymph = np.isin(names, LYMPHOID)
        core = lymph & (relative >= 0.5)
        rim = lymph & (relative < 0.5)
        b_core = float((names[core] == "B cell").mean()) if core.any() else np.nan
        b_rim = float((names[rim] == "B cell").mean()) if rim.any() else np.nan
        description = {
            "aggregate": aggregate_id,
            "n_B": n_b,
            "n_T": n_t,
            "B_share": n_b / max(n_b + n_t, 1),
            "B_share_core": b_core,
            "B_share_rim": b_rim,
            "B_core_index": b_core - b_rim if np.isfinite(b_core) and np.isfinite(b_rim) else np.nan,
            "max_depth_um": float(depth.max()),
        }
        for cell_type in ("epithelial", "macrophage", "CAF/fibro", "endothelial"):
            description[f"share_{cell_type}"] = float((names == cell_type).mean())
        if lv4 is not None:
            lv4_names = lv4.cells["label"].reindex(data.cells.index).to_numpy()
            inside_names = lv4_names[members]
            ring = polygon.buffer(ring_um).difference(polygon)
            x0, y0, x1, y1 = ring.bounds
            candidates = np.asarray(
                tree.query_ball_point([(x0 + x1) / 2, (y0 + y1) / 2], r=np.hypot(x1 - x0, y1 - y0) / 2 + 1), dtype=int
            )
            ring_members = candidates[shapely.contains_xy(ring, xy[candidates, 0], xy[candidates, 1])]
            ring_names = lv4_names[ring_members]
            for cell_type in ("plasma cell", "T CD4", "T CD8", "B cell"):
                description[f"lv4_{cell_type}_inside"] = float((inside_names == cell_type).mean())
                description[f"lv4_{cell_type}_ring"] = (
                    float((ring_names == cell_type).mean()) if len(ring_names) else np.nan
                )
        rows.append(description)

    return pd.DataFrame(rows).set_index("aggregate") if rows else pd.DataFrame()


def zoning(aggregates: pd.DataFrame, data: SampleData, k: int = 6) -> pd.Series:
    """
    How B and T cells are arranged inside each aggregate: the share of B-T pairs among the
    nearest-neighbour pairs of its lymphocytes, divided by the share expected if B and T were
    mixed like salt and pepper (2 pB pT). 1: mixed; towards 0: separate B and T zones, the
    organisation of a follicle and its T zone. A spot only gives the two proportions, never this.
    """

    labels = data.cells["label"]
    xy = data.cells[["x_um", "y_um"]]
    out = {}
    for aggregate_id, row in aggregates.iterrows():
        names = labels.loc[row["cell_ids"]]
        lymph = names[names.isin(LYMPHOID)]
        if len(lymph) < 20 or lymph.nunique() < 2:
            out[aggregate_id] = np.nan
            continue
        points = xy.loc[lymph.index].to_numpy()
        _, neighbours = cKDTree(points).query(points, k=k + 1)
        is_b = lymph.to_numpy() == "B cell"
        pairs_b = is_b[neighbours[:, 1:]]
        heterotypic = (pairs_b != is_b[:, None]).mean()
        p_b = is_b.mean()
        out[aggregate_id] = float(heterotypic / (2 * p_b * (1 - p_b)))

    return pd.Series(out, name="BT_mixing")


def follicles(
    aggregates: pd.DataFrame,
    data: SampleData,
    eps_um: float = 12.0,
    min_samples: int = 8,
    min_b: int = 30,
    min_b_share: float = 0.7,
) -> pd.DataFrame:
    """
    B-cell follicles inside the aggregates: dense clusters of B cells (DBSCAN) where B cells
    make up at least ``min_b_share`` of the lymphocytes.
    """

    labels = data.cells["label"]
    xy = data.cells[["x_um", "y_um"]]
    rows = []
    for aggregate_id, row in aggregates.iterrows():
        names = labels.loc[row["cell_ids"]]
        b_cells = names.index[names == "B cell"]
        if len(b_cells) < min_b:
            continue
        points = xy.loc[b_cells].to_numpy()
        clusters = DBSCAN(eps=eps_um, min_samples=min_samples).fit_predict(points)
        lymph_ids = names.index[names.isin(LYMPHOID)]
        lymph_xy = xy.loc[lymph_ids].to_numpy()
        lymph_b = names.loc[lymph_ids].to_numpy() == "B cell"
        for cluster in np.unique(clusters[clusters >= 0]):
            members = points[clusters == cluster]
            if len(members) < min_b:
                continue
            polygon = shapely.concave_hull(MultiPoint(members), ratio=0.3).buffer(5.0)
            inside = shapely.contains_xy(polygon, lymph_xy[:, 0], lymph_xy[:, 1])
            share = float(lymph_b[inside].mean()) if inside.any() else 0.0
            if share < min_b_share:
                continue
            rows.append(
                {
                    "aggregate_id": aggregate_id,
                    "polygon_um": polygon,
                    "n_B": int(len(members)),
                    "B_share": share,
                    "area_um2": float(polygon.area),
                }
            )

    return pd.DataFrame(rows)


def radial_profiles(aggregates: pd.DataFrame, data: SampleData, bins: int = 5) -> pd.DataFrame:
    """B share among the lymphoid nuclei by relative depth (0 = rim, 1 = deepest point), per aggregate."""

    labels = data.cells["label"]
    xy = data.cells[["x_um", "y_um"]]
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for aggregate_id, row in aggregates.iterrows():
        polygon = row["polygon_um"]
        names = labels.loc[row["cell_ids"]].to_numpy()
        points = xy.loc[row["cell_ids"]].to_numpy()
        depth = np.array([polygon.exterior.distance(Point(p)) for p in points])
        relative = depth / depth.max() if depth.max() > 0 else depth
        lymph = np.isin(names, LYMPHOID)
        for i in range(bins):
            selection = (
                lymph
                & (relative >= edges[i])
                & (relative <= edges[i + 1] if i == bins - 1 else relative < edges[i + 1])
            )
            if selection.sum() >= 5:
                rows.append(
                    {
                        "aggregate": aggregate_id,
                        "depth_bin": (edges[i] + edges[i + 1]) / 2,
                        "B_share": float((names[selection] == "B cell").mean()),
                        "n": int(selection.sum()),
                    }
                )

    return pd.DataFrame(rows)


def spot_region_profiles(regions: pd.DataFrame, data: SampleData) -> pd.DataFrame:
    """For the spot regions with an inside: B share of B + T in their edge spots and their inner spots."""

    neighbours = data.spot_neighbours()
    rows = []
    for region_id, row in regions.iterrows():
        members = set(row["spots"])
        inner = [s for s in members if len(neighbours[s]) >= 6 and all(n in members for n in neighbours[s])]
        edge = [s for s in members if s not in inner]
        if not inner:
            continue
        for where, spots in (("edge spots", edge), ("inner spots", inner)):
            p = data.props.loc[spots]
            rows.append(
                {
                    "region": region_id,
                    "where": where,
                    "B_share": float((p["B cell"] / (p["B cell"] + p["T cell"])).mean()),
                    "n_spots": len(spots),
                }
            )

    return pd.DataFrame(rows)


def to_geojson(tables: Dict[str, pd.DataFrame], data: SampleData, colors: Dict[str, Sequence[int]]) -> dict:
    """The aggregates of several methods as QuPath annotations (level-0 pixel coordinates)."""

    features = []
    for method, table in tables.items():
        for aggregate_id, row in table.iterrows():
            polygon = shapely.affinity.scale(row["polygon_um"], xfact=1 / data.mpp, yfact=1 / data.mpp, origin=(0, 0))
            geometry = shapely.geometry.mapping(polygon)
            measurements = [{"name": "area (µm²)", "value": round(float(row["area_um2"]), 1)}]
            for key in ("n_cells", "purity", "n_spots", "mean_score"):
                if key in row and pd.notna(row[key]):
                    measurements.append({"name": key, "value": round(float(row[key]), 4)})
            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        "objectType": "annotation",
                        "name": aggregate_id,
                        "classification": {"name": method, "colorRGB": _rgb_int(colors[method])},
                        "measurements": measurements,
                    },
                }
            )

    return {"type": "FeatureCollection", "features": features}


def _rgb_int(rgb: Sequence[int]) -> int:
    r, g, b = (int(v) for v in rgb[:3])
    return -(((255 - r) << 16) + ((255 - g) << 8) + (255 - b)) - 1


def polygon_px(polygon_um, mpp: float):
    return shapely.affinity.scale(polygon_um, xfact=1 / mpp, yfact=1 / mpp, origin=(0, 0))


def draw_polygon(
    ax, polygon_um, mpp: float, color, linewidth: float = 1.5, fill_alpha: float = 0.0, linestyle="-"
) -> None:
    from matplotlib.patches import Polygon as PolygonPatch

    polygon = polygon_px(polygon_um, mpp)
    parts: List = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
    for part in parts:
        coordinates = np.asarray(part.exterior.coords)
        if fill_alpha > 0:
            ax.add_patch(PolygonPatch(coordinates, closed=True, facecolor=color, alpha=fill_alpha, edgecolor="none"))
        ax.add_patch(
            PolygonPatch(
                coordinates, closed=True, fill=False, edgecolor=color, linewidth=linewidth, linestyle=linestyle
            )
        )
