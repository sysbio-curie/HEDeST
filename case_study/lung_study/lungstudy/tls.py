"""
Lymphoid aggregates (candidate tertiary lymphoid structures), from cells and from spots.

From the HEDeST cells (``detect_cell_aggregates``):

1. a nucleus is lymphoid when its HEDeST label (the most probable type, mean of the 5 seeds,
   without PPSA, lv3) is B cell or T cell;
2. positions in µm;
3. DBSCAN on the lymphoid nuclei: eps = 15 µm, min_samples = 10 (at least 10 lymphoid
   nuclei within 15 µm, i.e. one per ~70 µm²: dense packing). Planned at 20 µm; on
   A_AAAA1879, 20 µm chained the peritumoral infiltrates into rings around the tumour nests,
   so it was tightened to 15 µm before any comparison;
4. a cluster is kept with at least 50 lymphoid nuclei and a purity of at least 60% (lymphoid
   nuclei among all nuclei inside its outline);
5. outline: concave hull of its nuclei, grown by 5 µm (a nucleus radius);
6. described by its size, its B:T balance, how B and T cells are arranged inside it (mixed or
   in separate zones: ``compartments``), its B-cell follicles, and the plasma cells around it (lv4).

**B and T zones** (``compartments``): inside an aggregate, the touching pairs of lymphocytes
(:mod:`lungstudy.contacts`) are B–B, T–T or B–T. If B and T cells were mixed like salt and
pepper, the share of B–T pairs would be 2·pB·pT; the *B–T contact ratio* is the observed share
over that (1: mixed; below 1: B cells touch B cells and T cells touch T cells, i.e. separate
zones), with a permutation p-value (B / T labels shuffled among the aggregate's lymphocytes).
The *rim and core* B shares are those of the lymphocytes in the outer and inner third of the
depth (distance to the outline over the aggregate's largest depth). The B lineage is used
throughout: lv3's "B cell" holds the plasma cells, which lv4 separates.

From the spots (``detect_spot_regions``), what a deconvolution allows: spots whose B + T
proportion reaches a threshold, grouped with their neighbours into regions; a region's
smallest unit is one spot, and it has no inside. The threshold is ``SPOT_THRESHOLD`` = 0.9
(lv3, the same B + T as the cells): among 0.1-0.9, the one whose regions coincide best with the
HEDeST aggregates and with the small-round reference on every slide (``threshold_sweep``), i.e.
the most favourable setting for the deconvolution. With the 10%-filtered proportions no spot
lies between 0.9 and 1 (every other type below 10% is set to 0), so 0.9 keeps the spots made of
B and T cells only, and any higher threshold gives the same regions.

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
DEFAULTS = dict(eps_um=15.0, min_samples=10, min_cells=50, min_purity=0.6)
SPOT_THRESHOLD = 0.9  # lv3 B + T of a spot: the outlines closest to HEDeST's and to the reference (``threshold_sweep``)
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SMALL_ROUND = dict(max_area_um2=25.0, min_circularity=0.8)


def lymphoid_cells(data: SampleData, types: Sequence[str] = LYMPHOID) -> pd.Series:
    """1 for the nuclei whose HEDeST label is one of ``types`` (B cell or T cell), else 0."""

    return data.cells["label"].isin(list(types)).astype(float)


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
    data: SampleData, types: Sequence[str] = LYMPHOID, threshold: float = SPOT_THRESHOLD, min_spots: int = 1
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


def union_iou(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """Intersection over union of the areas covered by two sets of outlines."""

    if not len(left) or not len(right):
        return 0.0
    a = shapely.unary_union(list(left["polygon_um"]))
    b = shapely.unary_union(list(right["polygon_um"]))
    union = a.union(b).area
    return float(a.intersection(b).area / union) if union else 0.0


def recall(found: pd.DataFrame, reference: pd.DataFrame, min_share: float = 0.2) -> float:
    """
    Share of the ``reference`` outlines touched by one of ``found``: their intersection covers at
    least ``min_share`` of the smaller of the two, so that a coarse region holding a small
    aggregate counts as finding it.
    """

    if not len(reference):
        return np.nan
    if not len(found):
        return 0.0
    tree = shapely.STRtree(list(found["polygon_um"]))
    touched = 0
    for polygon in reference["polygon_um"]:
        for j in tree.query(polygon, predicate="intersects"):
            other = found["polygon_um"].iloc[j]
            if polygon.intersection(other).area >= min_share * min(polygon.area, other.area):
                touched += 1
                break

    return touched / len(reference)


def threshold_sweep(
    data: SampleData, aggregates: pd.DataFrame, reference: pd.DataFrame, thresholds: Sequence[float] = THRESHOLDS
) -> pd.DataFrame:
    """
    The spot regions at every B + T threshold (lv3, ``detect_spot_regions``), against the HEDeST
    aggregates and the small-round-nuclei reference: number of regions, area, share of the HEDeST
    aggregates they touch, and how well the covered areas coincide (IoU).
    """

    rows = []
    for threshold in thresholds:
        regions = detect_spot_regions(data, threshold=threshold)
        rows.append(
            {
                "sample": data.name,
                "threshold": threshold,
                "n_regions": len(regions),
                "area_mm2": float(regions["area_um2"].sum() / 1e6) if len(regions) else 0.0,
                "recall": recall(regions, aggregates),
                "iou": union_iou(regions, aggregates),
                "iou_reference": union_iou(regions, reference),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ description


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


# ------------------------------------------------------------------ B and T zones


def lineage_codes(labels: pd.Series) -> np.ndarray:
    """1 for the B lineage (B and plasma cells), 0 for T cells (any subtype), -1 otherwise."""

    from lungstudy.style import T_SUBTYPES

    names = labels.to_numpy()
    codes = np.full(len(names), -1, dtype=np.int8)
    codes[np.isin(names, ["B cell", "plasma cell"])] = 1
    codes[np.isin(names, ["T cell", *T_SUBTYPES])] = 0

    return codes


def depth_in(polygon, xy: np.ndarray) -> np.ndarray:
    """Distance (µm) of points to the outline of a polygon (the points are inside it)."""

    return shapely.distance(polygon.boundary, shapely.points(xy[:, 0], xy[:, 1]))


def compartments(
    aggregates: pd.DataFrame,
    data: SampleData,
    labels: Optional[pd.Series] = None,
    n_permutations: int = 200,
    seed: int = 0,
    min_lymphocytes: int = 50,
) -> pd.DataFrame:
    """
    How B and T cells are arranged inside each aggregate: B share, B–T contact ratio and its
    permutation p-value, B share at the rim (relative depth < 1/3) and in the core (> 2/3),
    their difference (``B_core_minus_rim``), and the largest depth.
    """

    labels = data.cells["label"] if labels is None else labels.reindex(data.cells.index)
    codes = lineage_codes(labels)
    xy = data.cells[["x_um", "y_um"]].to_numpy()
    position = pd.Series(np.arange(len(data.cells)), index=data.cells.index)
    pairs = data.contacts
    rng = np.random.default_rng(seed)
    rows = []
    for aggregate_id, row in aggregates.iterrows():
        members = position.loc[row["cell_ids"]].to_numpy()
        lymph = members[codes[members] >= 0]
        record = {"aggregate": aggregate_id, "n_lymphocytes": len(lymph)}
        if len(lymph) < min_lymphocytes:
            rows.append(record)
            continue
        is_b = codes[lymph] == 1
        p_b = float(is_b.mean())
        local = np.full(len(data.cells), -1, dtype=np.int64)
        local[lymph] = np.arange(len(lymph))
        inner = pairs[(local[pairs[:, 0]] >= 0) & (local[pairs[:, 1]] >= 0)]
        a, b = local[inner[:, 0]], local[inner[:, 1]]
        observed = float(np.mean(is_b[a] != is_b[b])) if len(inner) else np.nan
        expected = 2 * p_b * (1 - p_b)
        if len(inner) and 0 < p_b < 1:
            null = np.empty(n_permutations)
            for k in range(n_permutations):
                shuffled = rng.permutation(is_b)
                null[k] = np.mean(shuffled[a] != shuffled[b])
            p_value = float((np.sum(null <= observed) + 1) / (n_permutations + 1))
            ratio = observed / null.mean()
        else:
            p_value, ratio = np.nan, np.nan
        depth = depth_in(row["polygon_um"], xy[lymph])
        relative = depth / depth.max() if depth.max() > 0 else depth
        rim, core = relative < 1 / 3, relative > 2 / 3
        record.update(
            {
                "B_share": p_b,
                "BT_pairs": int(len(inner)),
                "BT_contact_ratio": ratio,
                "BT_contact_p": p_value,
                "BT_expected_share": expected,
                "B_share_rim": float(is_b[rim].mean()) if rim.any() else np.nan,
                "B_share_core": float(is_b[core].mean()) if core.any() else np.nan,
                "max_depth_um": float(depth.max()),
            }
        )
        record["B_core_minus_rim"] = record["B_share_core"] - record["B_share_rim"]
        rows.append(record)

    return pd.DataFrame(rows).set_index("aggregate")


def zone_agreement(
    aggregates: pd.DataFrame,
    data: SampleData,
    others: Dict[str, pd.Series],
    radius_um: float = 25.0,
    min_lymphocytes: int = 200,
) -> pd.DataFrame:
    """
    Do independently trained models draw the same B and T zones? For each aggregate, the B
    share among the lymphocytes within ``radius_um`` of each lymphocyte (lv3), and the same
    with each other model's labels; the Spearman correlation of the two maps over the
    lymphocytes. Single-cell calls can differ while the zones agree.
    """

    from scipy.stats import spearmanr

    codes = {"main": lineage_codes(data.cells["label"])}
    codes.update({name: lineage_codes(labels.reindex(data.cells.index)) for name, labels in others.items()})
    xy = data.cells[["x_um", "y_um"]].to_numpy()
    position = pd.Series(np.arange(len(data.cells)), index=data.cells.index)
    rows = []
    for aggregate_id, row in aggregates.iterrows():
        members = position.loc[row["cell_ids"]].to_numpy()
        lymph = members[codes["main"][members] >= 0]
        if len(lymph) < min_lymphocytes:
            continue
        neighbours = cKDTree(xy[lymph]).query_ball_point(xy[lymph], radius_um)
        maps = {}
        for name, c in codes.items():
            is_b = (c[lymph] == 1).astype(float)
            is_l = (c[lymph] >= 0).astype(float)
            num = np.array([is_b[n].sum() for n in neighbours])
            den = np.array([is_l[n].sum() for n in neighbours])
            maps[name] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        record = {"aggregate": aggregate_id, "n_lymphocytes": len(lymph), "zone_sd": float(np.nanstd(maps["main"]))}
        for name in others:
            ok = np.isfinite(maps["main"]) & np.isfinite(maps[name])
            record[f"rho_{name}"] = float(spearmanr(maps["main"][ok], maps[name][ok])[0]) if ok.sum() > 20 else np.nan
        rows.append(record)

    return pd.DataFrame(rows).set_index("aggregate") if rows else pd.DataFrame()


def stage(row: pd.Series, max_ratio: float = 0.8, alpha: float = 0.01) -> str:
    """
    The organisation of an aggregate, from the cells alone:

    - *with B follicle*: holds at least one B follicle (a dense B core, see :func:`follicles`);
    - *B / T zones*: no follicle, but B and T touch each other at least 20% less than if mixed
      (B–T contact ratio ≤ 0.8, permutation p < 0.01);
    - *mixed*: neither.

    These are the first steps of TLS maturation (aggregation, then separate B and T zones, then a
    follicle); the later steps (follicular dendritic cells, germinal centre, high endothelial
    venules) are not visible to cell types learnt from this reference, and are looked for in
    the marker genes.
    """

    if row.get("n_follicles", 0) > 0:
        return "with B follicle"
    if row.get("BT_contact_ratio", np.nan) <= max_ratio and row.get("BT_contact_p", 1.0) < alpha:
        return "B / T zones"
    return "mixed"


STAGES = ["mixed", "B / T zones", "with B follicle"]
