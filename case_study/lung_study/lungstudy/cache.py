"""
Per-sample tables extracted once from the LungScape outputs, so the notebooks load in seconds.

``build(sample)`` reads the segmentation (hundreds of MB of JSON), the AnnData, the HEDeST
seed runs of every level and the proportion files, and writes under ``cache/<sample>/``:

- ``cells.pkl``: one row per nucleus: position (px and µm), nucleus shape, HoVer-Net type,
  whether it lies in a spot (and which), its nearest spot;
- ``contours.pkl``: the nucleus contours, for the slide views;
- ``pred_<level>[_raw].pkl``: the mean of the 5 seeds' predictions **without PPSA**;
- ``labels.pkl``: per level, the predicted type, its probability and the seed agreement
  (computed on the predictions without PPSA);
- ``props_<level>[_raw].pkl``: the proportions each run was trained on (normalised);
- ``spots.pkl``: one row per spot: centre, number of cells;
- ``seed_spots_lv3.pkl``: per seed, the mean prediction of every spot and the split it was in
  (train / val / test), rebuilt exactly as HEDeST drew it;
- ``meta.json``: mpp, spot diameter and pitch in pixels, the input paths.
"""
from __future__ import annotations

import json
import pickle
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from lungstudy import paths
from scipy.spatial import cKDTree

# (level, proportions) pairs cached for every sample: the four levels on the main
# proportions, plus the main level on the raw proportions for the sensitivity checks.
UNITS = [(level, paths.PROPORTIONS) for level in paths.LEVELS] + [(paths.MAIN_LEVEL, "raw")]


def unit_key(level: str, kind: str = paths.PROPORTIONS) -> str:
    """The cache file suffix of a unit."""

    return level if kind == paths.PROPORTIONS else f"{level}_{kind}"


# ------------------------------------------------------------------ building


def _shape_features(contours: List[np.ndarray]) -> pd.DataFrame:
    """Area, perimeter, circularity, eccentricity and solidity of every contour (in px)."""

    import cv2

    rows = np.zeros((len(contours), 5), dtype="float64")
    for index, contour in enumerate(contours):
        points = contour.astype(np.float32).reshape(-1, 1, 2)
        area = abs(cv2.contourArea(points))
        perimeter = cv2.arcLength(points, closed=True)
        moments = cv2.moments(points)
        eccentricity = np.nan
        if moments["m00"] > 0:
            mu20, mu02, mu11 = moments["mu20"], moments["mu02"], moments["mu11"]
            common = np.sqrt(4 * mu11**2 + (mu20 - mu02) ** 2)
            major, minor = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
            if major > 0:
                eccentricity = np.sqrt(max(0.0, 1 - minor / major))
        hull_area = abs(cv2.contourArea(cv2.convexHull(points)))
        rows[index] = (
            area,
            perimeter,
            4 * np.pi * area / perimeter**2 if perimeter > 0 else np.nan,
            eccentricity,
            area / hull_area if hull_area > 0 else np.nan,
        )

    return pd.DataFrame(rows, columns=["area_px", "perimeter_px", "circularity", "eccentricity", "solidity"])


def build(name: str, overwrite: bool = False) -> None:
    """Builds the cache of one sample."""

    import anndata as ad

    from hedest.analysis.loaders import load_seed_runs
    from hedest.analysis.seeds import SeedEnsemble
    from hedest.dataset_utils import split_data
    from hedest.spots import spot_geometry

    sample = paths.sample(name)
    out = sample.cache()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists() and not overwrite:
        logger.info(f"{name}: cache exists, skipped")
        return

    mpp = sample.mpp

    # ---- segmentation
    logger.info(f"{name}: reading {sample.segmentation}")
    with open(sample.segmentation) as handle:
        nuc = json.load(handle)["nuc"]
    cell_ids = list(nuc)
    centroids = np.asarray([nuc[c]["centroid"] for c in cell_ids], dtype="float64")
    contours = [np.asarray(nuc[c]["contour"], dtype=np.int32) for c in cell_ids]
    hv_type = np.asarray([int(nuc[c].get("type") or 0) for c in cell_ids])
    hv_prob = np.asarray([float(nuc[c].get("type_prob") or np.nan) for c in cell_ids])
    del nuc

    shapes = _shape_features(contours)
    cells = pd.DataFrame(
        {
            "x": centroids[:, 0],
            "y": centroids[:, 1],
            "x_um": centroids[:, 0] * mpp,
            "y_um": centroids[:, 1] * mpp,
            "area_um2": shapes["area_px"].to_numpy() * mpp**2,
            "perimeter_um": shapes["perimeter_px"].to_numpy() * mpp,
            "circularity": shapes["circularity"].to_numpy(),
            "eccentricity": shapes["eccentricity"].to_numpy(),
            "solidity": shapes["solidity"].to_numpy(),
            "hv_type": hv_type,
            "hv_name": [paths.HOVERNET_TYPES.get(t, "?") for t in hv_type],
            "hv_prob": hv_prob,
        },
        index=pd.Index(cell_ids, name="cell_id"),
    )
    with open(out / "contours.pkl", "wb") as handle:
        pickle.dump({"ids": cell_ids, "contours": contours}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- spots
    adata = ad.read_h5ad(sample.adata)
    adata_name = list(adata.uns["spatial"])[0]
    geometry = spot_geometry(adata, adata_name, mpp=mpp)
    spot_ids = [str(s) for s in geometry.ids]
    centers = np.asarray(geometry.centers[:, :2], dtype="float64")
    pitch_px = float(np.median(cKDTree(centers).query(centers, k=2)[0][:, 1]))

    labels: Dict[str, pd.Series] = {}
    spot_dict: Optional[Dict[str, List[str]]] = None
    for level, kind in UNITS:
        key = unit_key(level, kind)
        run_dir = sample.run_dir(level, kind)
        logger.info(f"{name}: {key} from {run_dir}")
        runs = load_seed_runs(run_dir, adjusted=False)
        if len(runs) != len(paths.SEEDS):
            raise RuntimeError(f"{run_dir}: {len(runs)} seeds found, {len(paths.SEEDS)} expected")
        ensemble = SeedEnsemble(runs, adjusted=False)
        mean = ensemble.mean().reindex(cells.index).astype("float32")
        mean.to_pickle(out / f"pred_{key}.pkl")
        runs[0].proportions.to_pickle(out / f"props_{key}.pkl")

        labels[f"label_{key}"] = mean.idxmax(axis=1)
        labels[f"conf_{key}"] = mean.max(axis=1)
        labels[f"agree_{key}"] = ensemble.agreement().reindex(cells.index)

        if (level, kind) == (paths.MAIN_LEVEL, paths.PROPORTIONS):
            spot_dict = runs[0].spot_dict
            rows = []
            for run in runs:
                params = run.params
                order = list(run.spot_dict)
                train, _, val, _, test, _ = split_data(
                    run.spot_dict,
                    run.proportions.reindex(order),
                    train_size=params["train_size"],
                    val_size=params["val_size"],
                    rs=params["rs"],
                )
                if set(train) != set(run.train_spot_dict or {}):
                    raise RuntimeError(f"{run.run_dir}: the rebuilt training split differs from the saved one")
                split = {**{s: "train" for s in train}, **{s: "val" for s in val}, **{s: "test" for s in test}}
                predictions = run.predictions_raw
                for spot, members in run.spot_dict.items():
                    members = [m for m in members if m in predictions.index]
                    if not members:
                        continue
                    row = predictions.loc[members].mean(axis=0)
                    rows.append({"seed": params["rs"], "spot": spot, "split": split[spot], **row.to_dict()})
            pd.DataFrame(rows).to_pickle(out / "seed_spots_lv3.pkl")

    pd.DataFrame(labels).to_pickle(out / "labels.pkl")

    # ---- cells and spots
    owner = {cell: spot for spot, members in (spot_dict or {}).items() for cell in members}
    cells["spot"] = [owner.get(c, "") for c in cells.index]
    cells["in_spot"] = cells["spot"] != ""
    distance, nearest = cKDTree(centers).query(centroids)
    cells["nearest_spot"] = np.asarray(spot_ids)[nearest]
    cells["nearest_spot_dist_um"] = distance * mpp
    cells.to_pickle(out / "cells.pkl")

    counts = cells.loc[cells["in_spot"], "spot"].value_counts()
    spots = pd.DataFrame(
        {"x": centers[:, 0], "y": centers[:, 1], "x_um": centers[:, 0] * mpp, "y_um": centers[:, 1] * mpp},
        index=pd.Index(spot_ids, name="spot"),
    )
    spots["n_cells"] = counts.reindex(spots.index).fillna(0).astype(int)
    spots.to_pickle(out / "spots.pkl")

    meta = {
        "sample": name,
        "mpp": mpp,
        "spot_diameter_px": float(geometry.diameter),
        "spot_pitch_px": pitch_px,
        "spot_pitch_um": pitch_px * mpp,
        "slide": str(sample.slide),
        "segmentation": str(sample.segmentation),
        "image_dict": str(sample.image_dict),
        "adata": str(sample.adata),
        "adata_name": adata_name,
        "n_cells": int(len(cells)),
        "n_spots": int(len(spots)),
    }
    with open(out / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)
    logger.info(f"{name}: cache written to {out}")


# ------------------------------------------------------------------ loading


def meta(name: str) -> dict:
    with open(paths.sample(name).cache() / "meta.json") as handle:
        return json.load(handle)


def cells(name: str) -> pd.DataFrame:
    return pd.read_pickle(paths.sample(name).cache() / "cells.pkl")


def spots(name: str) -> pd.DataFrame:
    return pd.read_pickle(paths.sample(name).cache() / "spots.pkl")


def labels(name: str) -> pd.DataFrame:
    return pd.read_pickle(paths.sample(name).cache() / "labels.pkl")


def predictions(name: str, level: str = paths.MAIN_LEVEL, kind: str = paths.PROPORTIONS) -> pd.DataFrame:
    return pd.read_pickle(paths.sample(name).cache() / f"pred_{unit_key(level, kind)}.pkl")


def proportions(name: str, level: str = paths.MAIN_LEVEL, kind: str = paths.PROPORTIONS) -> pd.DataFrame:
    """The proportions the HEDeST run was trained on (rows normalised to 1, as HEDeST does)."""

    return pd.read_pickle(paths.sample(name).cache() / f"props_{unit_key(level, kind)}.pkl")


def seed_spots(name: str) -> pd.DataFrame:
    return pd.read_pickle(paths.sample(name).cache() / "seed_spots_lv3.pkl")


def contours(name: str) -> Dict[str, np.ndarray]:
    with open(paths.sample(name).cache() / "contours.pkl", "rb") as handle:
        payload = pickle.load(handle)
    return dict(zip(payload["ids"], payload["contours"]))


def table(name: str, level: str = paths.MAIN_LEVEL, kind: str = paths.PROPORTIONS) -> pd.DataFrame:
    """The cells joined with the label, probability and agreement of one unit."""

    key = unit_key(level, kind)
    frame = cells(name).join(labels(name)[[f"label_{key}", f"conf_{key}", f"agree_{key}"]])

    return frame.rename(columns={f"label_{key}": "label", f"conf_{key}": "conf", f"agree_{key}": "agree"})
