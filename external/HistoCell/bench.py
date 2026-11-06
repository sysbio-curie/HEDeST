"""I/O adapters between the STHELAR benchmark layout and HistoCell.

Upstream HistoCell reads a directory of pre-cut ``.jpg``/``.png`` tiles, one
HoVer-Net ``.json`` per tile, and one ``.tsv`` of spot proportions per sample.
The benchmark instead stores, per slide:

===========================  ====================================================
``he.tiff``                  pyramidal RGB whole-slide image
``pseudovisium.h5ad``        spot table; ``obsm['spatial']`` = (x, y) in he.tiff px
``hovernet.json``            one HoVer-Net dict for the *whole slide*
``sim/level{L}/proportions.csv``  spot x cell-type proportions
===========================  ====================================================

This module turns the second layout into the first *in memory* -- tiles are cut
on the fly and nuclei are re-expressed in tile coordinates -- so that the model,
the losses and every hyper-parameter stay exactly as released.  Nothing here
touches the method; see README.md ("Benchmark adaptation") for the full list.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whole-slide image
# ---------------------------------------------------------------------------
class SlideReader:
    """Full-resolution RGB pixels of a whole-slide TIFF, held in memory.

    The slides here are 1-4 GB decoded and every nucleus is read exactly once,
    so a single decode into a plain array beats tile-wise random access.  The
    array is created before the DataLoader workers fork, so on Linux the
    workers share it copy-on-write rather than each holding a copy.
    """

    def __init__(self, path: str):
        import tifffile

        self.path = path
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            self.array = series.asarray()
        if self.array.ndim == 2:  # grayscale -> RGB
            self.array = np.repeat(self.array[:, :, None], 3, axis=2)
        if self.array.shape[2] == 4:  # RGBA -> RGB
            self.array = self.array[:, :, :3]
        self.height, self.width = self.array.shape[:2]
        logger.info(
            "Loaded H&E %s (%d x %d, %.2f GB in memory)",
            path,
            self.width,
            self.height,
            self.array.nbytes / 1e9,
        )

    def crop(self, x0: int, y0: int, size: int) -> np.ndarray:
        """Crop ``size`` x ``size`` at (x0, y0); truncated at the slide border.

        Upstream's ``create_tile`` clamps the top-left corner to 0 and lets
        numpy truncate at the far edge, so a border tile comes out smaller and
        is stretched to 256 x 256 all the same.  Same behaviour here.
        """
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        return self.array[y0 : y0 + size, x0 : x0 + size]


# ---------------------------------------------------------------------------
# Segmentation (HoVer-Net dict)
# ---------------------------------------------------------------------------
def load_seg_dict(path: str) -> Tuple[pd.DataFrame, dict]:
    """Read a whole-slide HoVer-Net ``.json`` into a nucleus table.

    Returns ``(nuclei, meta)`` where ``nuclei`` is indexed by the *original*
    cell id -- the id the benchmark uses everywhere else -- with columns

        cx, cy   centroid, whole-slide pixels ("centroid": [x, y])
        r0, c0   bbox top-left, ``[[rmin, cmin], [rmax, cmax]]`` convention
        r1, c1   bbox bottom-right
        htype    PanNuke class index (0 nolabel, 1 neoplastic, 2 inflammatory,
                 3 connective, 4 dead, 5 epithelial)

    The bounding box is **recomputed from the contour** rather than read from
    the file's ``bbox`` field.  In this benchmark's ``hovernet.json`` files the
    stored ``bbox`` is the contour's box with the two per-tile offsets
    exchanged -- it comes out as ``[[ymin - K, xmin + K], [ymax - K, xmax + K]]``
    with ``K = (tile_row - tile_col) * tile_step`` -- so it only lands on the
    nucleus for the few percent of cells sitting on the tile diagonal, where
    K is 0.  ``centroid`` and ``contour`` are consistent with each other and
    with the image; deriving the box from the contour is what HoVer-Net's own
    ``get_bounding_box`` computes, and it is what PanoSpace's adapter does too.
    The disagreement rate is measured and logged on every run.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    nuc = raw["nuc"] if "nuc" in raw else raw
    meta = {k: v for k, v in raw.items() if k != "nuc"}

    ids, cx, cy, r0, c0, r1, c1, htype = [], [], [], [], [], [], [], []
    n_stored_wrong, n_no_contour = 0, 0
    for cid, entry in nuc.items():
        x, y = float(entry["centroid"][0]), float(entry["centroid"][1])
        contour = entry.get("contour")
        if contour:
            xs = [p[0] for p in contour]
            ys = [p[1] for p in contour]
            a, b, c, d = min(ys), min(xs), max(ys) + 1, max(xs) + 1
        else:
            # No contour to fall back on: a box the size of a small nucleus,
            # centred on the centroid, so the cell still gets a crop.
            n_no_contour += 1
            a, b, c, d = int(y) - 8, int(x) - 8, int(y) + 8, int(x) + 8
        if "bbox" in entry:
            (sa, sb), _ = entry["bbox"]
            if int(sa) != int(a) or int(sb) != int(b):
                n_stored_wrong += 1

        ids.append(str(cid))
        cx.append(x)
        cy.append(y)
        r0.append(int(a))
        c0.append(int(b))
        r1.append(int(c))
        c1.append(int(d))
        t = entry.get("type")
        htype.append(-1 if t is None else int(t))

    nuclei = pd.DataFrame(
        {"cx": cx, "cy": cy, "r0": r0, "c0": c0, "r1": r1, "c1": c1, "htype": htype},
        index=pd.Index(ids, name="cell_id"),
    )
    logger.info("Loaded segmentation: %d nuclei from %s", len(nuclei), path)
    if n_no_contour:
        logger.warning("%d nucleus/nuclei carry no contour; boxed around the centroid instead", n_no_contour)
    if n_stored_wrong:
        logger.warning(
            "%d/%d nuclei (%.1f%%) have a stored 'bbox' that does not match their own "
            "contour; using the contour. See load_seg_dict.__doc__ for why.",
            n_stored_wrong,
            len(nuclei),
            100.0 * n_stored_wrong / len(nuclei),
        )
    return nuclei, meta


# ---------------------------------------------------------------------------
# Spot table
# ---------------------------------------------------------------------------
def load_st_adata(path: str):
    """Open the spot AnnData in backed mode -- only obs/obsm/uns are read."""
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    if "spatial" not in adata.obsm:
        raise KeyError(f"{path}: obsm['spatial'] (spot pixel coordinates) is required")
    logger.info("Loaded ST data %s (%d spots)", path, adata.n_obs)
    return adata


def spot_coordinates(adata) -> pd.DataFrame:
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    return pd.DataFrame(xy[:, :2], index=adata.obs_names.astype(str), columns=["x", "y"])


def infer_tile_size(adata) -> float:
    """Side of a HistoCell tile, in whole-slide pixels.

    The paper cuts tiles "according to the size of spots (e.g. 55 um for 10x
    Visium)", so the tile side is the spot *diameter*.  Taken from
    ``uns['spatial'][lib]['scalefactors']['spot_diameter_fullres']`` when the
    file provides it, then from ``uns['pseudovisium']``, and only failing both
    from the nearest-neighbour spot pitch.
    """
    spatial = adata.uns.get("spatial", {})
    for lib in spatial.values():
        sf = lib.get("scalefactors", {}) if isinstance(lib, dict) else {}
        if "spot_diameter_fullres" in sf:
            d = float(sf["spot_diameter_fullres"])
            logger.info("Tile size = spot diameter = %.1f px (scalefactors)", d)
            return d

    pv = adata.uns.get("pseudovisium", {})
    if "spot_diameter_um" in pv and "mpp" in pv:
        d = float(pv["spot_diameter_um"]) / float(pv["mpp"])
        logger.info("Tile size = spot diameter = %.1f px (%s um / %s mpp)", d, pv["spot_diameter_um"], pv["mpp"])
        return d

    from scipy.spatial import cKDTree

    xy = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    dist, _ = cKDTree(xy).query(xy, k=2)
    pitch = float(np.median(dist[:, 1]))
    logger.warning(
        "No spot diameter recorded; falling back to the spot pitch (%.1f px). " "Pass --tile-px to set it explicitly.",
        pitch,
    )
    return pitch


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------
def load_proportions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.index.name = "spot_id"
    logger.info(
        "Loaded proportions %s (%d spots x %d cell types: %s)", path, df.shape[0], df.shape[1], list(df.columns)
    )
    return df


def write_proportions(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.index.name = "spot_id"
    df.to_csv(path)
    logger.info("Wrote proportions (%d spots x %d cell types) -> %s", *df.shape, path)


def write_predictions(df: pd.DataFrame, path: str) -> None:
    """Write the cell x cell-type table, shaped like HEDeST ``pred_best_adjusted``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.index.name = "cell_id"
    df.to_csv(path)
    logger.info("Wrote predictions (%d cells x %d cell types) -> %s", *df.shape, path)


# ---------------------------------------------------------------------------
# Tissue compartment file (upstream ./tcs/*.json)
# ---------------------------------------------------------------------------
# Upstream maps the six PanNuke classes onto its three compartments with the
# same constant in every ./tcs/*.json.  Copied verbatim:
#   0 nolabel -> Stromal, 1 neoplastic -> Epi,  2 inflammatory -> TME,
#   3 connective -> Stromal, 4 dead -> Stromal, 5 epithelial -> Epi
PANNUKE_TO_COMPARTMENT = [2, 0, 1, 2, 2, 0]
COMPARTMENTS = ["Epi", "TME", "Stromal"]

# Some labels are too short to match on a substring without catching unrelated
# names ("B" is inside every other word), so they are resolved by exact name.
# All of these appear at levels 3-4 of the benchmark's hierarchies.
_COMPARTMENT_EXACT: Dict[str, str] = {
    "b": "TME",
    "t": "TME",
    "nk": "TME",
    "dc": "TME",
    "pdc": "TME",
    "cdc": "TME",
    "t_cd4": "TME",
    "t_cd8": "TME",
    "b_cell": "TME",
    "t_cell": "TME",
}

# Keyword rules used when no --tissue-compartment file is given.  Checked in
# this order, first match wins, so "Blood_vessel" reaches Stromal and
# "B_Plasma" reaches TME before either can hit a broader rule.
_COMPARTMENT_RULES: List[Tuple[str, Sequence[str]]] = [
    (
        "TME",
        (
            "immune",
            "lymphoid",
            "lympho",
            "myeloid",
            "t_nk",
            "tcell",
            "t_cell",
            "b_plasma",
            "bcell",
            "b_cell",
            "plasma",
            "macrophage",
            "mast",
            "dendritic",
            "neutrophil",
            "granulocyte",
            "monocyte",
            "microglia",
            "leukocyte",
            "nk_",
            "_nk",
            "dc_",
            "treg",
            "cd4",
            "cd8",
            "cd3",
        ),
    ),
    (
        "Stromal",
        (
            "stroma",
            "structural",
            "fibroblast",
            "myofibroblast",
            "caf",
            "endotheli",
            "blood_vessel",
            "vessel",
            "vascular",
            "pericyte",
            "smooth_muscle",
            "muscle",
            "adipocyte",
            "mesenchym",
            "neural",
            "glia",
            "schwann",
            "neuron",
            "lymphatic",
        ),
    ),
    (
        "Epi",
        (
            "epitheli",
            "tumor",
            "tumour",
            "cancer",
            "neoplastic",
            "melanocyte",
            "keratinocyte",
            "hepatocyte",
            "acinar",
            "ductal",
            "luminal",
            "basal",
            "secretory",
            "club",
            "ciliated",
            "alveolar",
            "at1",
            "at2",
            "goblet",
            "neuroendocrine",
            "hillock",
            "urothel",
        ),
    ),
]


def compartment_of(cell_type: str) -> Optional[str]:
    name = str(cell_type).strip().lower()
    if name in _COMPARTMENT_EXACT:
        return _COMPARTMENT_EXACT[name]
    for compartment, keys in _COMPARTMENT_RULES:
        if any(k in name for k in keys):
            return compartment
    return None


def make_tissue_compartment(cell_types: Sequence[str]) -> dict:
    """Build an upstream-shaped ``tcs`` dict for this level's vocabulary.

    Upstream ships one hand-written file per tissue under ``./tcs``.  The
    benchmark's vocabulary changes with the annotation level, so the same file
    is derived from the cell-type names instead -- the *contents* keep exactly
    upstream's meaning and are written next to the results for inspection.
    Unmatched names fall back to ``Stromal``, upstream's own catch-all for the
    PanNuke "nolabel" class, and are reported loudly.
    """
    mapping, unmatched = {}, []
    for i, ct in enumerate(cell_types):
        comp = compartment_of(ct)
        if comp is None:
            unmatched.append(str(ct))
            comp = "Stromal"
        mapping[str(i)] = comp
    if unmatched:
        logger.warning(
            "No compartment rule matched %s -- assigned to 'Stromal'. Pass " "--tissue-compartment to override.",
            unmatched,
        )
    tcs = {"dict": mapping, "list": list(COMPARTMENTS), "HoVerNet": list(PANNUKE_TO_COMPARTMENT)}
    logger.info("Tissue compartments: %s", {ct: mapping[str(i)] for i, ct in enumerate(cell_types)})
    return tcs


def load_tissue_compartment(path: str, n_cell_types: int) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        tcs = json.load(fh)
    for key in ("dict", "list", "HoVerNet"):
        if key not in tcs:
            raise KeyError(f"{path}: tissue-compartment file has no '{key}' entry")
    if len(tcs["dict"]) != n_cell_types:
        raise ValueError(
            f"{path}: 'dict' has {len(tcs['dict'])} entries but the proportions " f"table has {n_cell_types} cell types"
        )
    return tcs


def write_tissue_compartment(tcs: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(tcs, fh, indent=4)


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------
def spot_tiles(coords: pd.DataFrame, tile_px: float) -> pd.DataFrame:
    """One tile per spot, of side ``tile_px``, centred on the spot.

    Mirrors ``tutorial.ipynb::create_tile``: the top-left corner is clamped to
    0 and the far edge is left to the slide bounds.
    """
    half = tile_px / 2.0
    return pd.DataFrame(
        {
            "x0": np.maximum(0, np.round(coords["x"].to_numpy() - half)).astype(int),
            "y0": np.maximum(0, np.round(coords["y"].to_numpy() - half)).astype(int),
        },
        index=coords.index,
    )


def grid_tiles(nuclei: pd.DataFrame, tile_px: float, width: int, height: int) -> pd.DataFrame:
    """Partition the slide into ``tile_px`` tiles; keep those holding a nucleus.

    Inference has to reach *every* nucleus, not only the ones under a spot, and
    the spots cover well under half the tissue.  A regular partition at the
    training tile size is what the paper does for data without spots
    (equations 1-5, the Xenium grid), and it gives each nucleus exactly one
    tile, at exactly the magnification the model was trained on.
    """
    step = int(round(tile_px))
    ix = np.floor(nuclei["cx"].to_numpy() / step).astype(int)
    iy = np.floor(nuclei["cy"].to_numpy() / step).astype(int)
    keys = np.unique(np.stack([ix, iy], axis=1), axis=0)
    tiles = pd.DataFrame(
        {"x0": keys[:, 0] * step, "y0": keys[:, 1] * step},
        index=pd.Index([f"tile_{a:05d}x{b:05d}" for a, b in keys], name="tile_id"),
    )
    logger.info(
        "Inference grid: %d tiles of %d px covering %d nuclei " "(slide %d x %d)",
        len(tiles),
        step,
        len(nuclei),
        width,
        height,
    )
    return tiles


def assign_nuclei(
    nuclei: pd.DataFrame, tiles: pd.DataFrame, tile_px: float, restrict: Optional[Dict[str, Sequence[str]]] = None
) -> Dict[str, np.ndarray]:
    """Map each tile to the row numbers of the nuclei whose centroid is inside.

    ``restrict`` optionally limits a tile to a given list of cell ids (used by
    ``--spot-dict``); nuclei outside the tile square are still dropped, since
    their crop would not exist.
    """
    size = int(round(tile_px))
    positions = {cid: i for i, cid in enumerate(nuclei.index)}
    cx = nuclei["cx"].to_numpy()
    cy = nuclei["cy"].to_numpy()

    # Bucket nuclei by the grid cell of their centroid, then look up only the
    # (at most four) buckets a tile can overlap -- an N x M scan is far too slow
    # at 170k nuclei x 17k tiles.
    bucket: Dict[Tuple[int, int], List[int]] = {}
    for i, (a, b) in enumerate(zip(np.floor(cx / size).astype(int), np.floor(cy / size).astype(int))):
        bucket.setdefault((int(a), int(b)), []).append(i)

    out: Dict[str, np.ndarray] = {}
    for tid, (x0, y0) in zip(tiles.index, tiles[["x0", "y0"]].to_numpy()):
        cand: List[int] = []
        bx, by = int(np.floor(x0 / size)), int(np.floor(y0 / size))
        for a in (bx, bx + 1):
            for b in (by, by + 1):
                cand.extend(bucket.get((a, b), ()))
        if not cand:
            out[tid] = np.empty(0, dtype=int)
            continue
        cand_arr = np.asarray(sorted(set(cand)), dtype=int)
        inside = (cx[cand_arr] >= x0) & (cx[cand_arr] < x0 + size) & (cy[cand_arr] >= y0) & (cy[cand_arr] < y0 + size)
        sel = cand_arr[inside]
        if restrict is not None:
            allowed = {positions[c] for c in restrict.get(tid, ()) if c in positions}
            sel = np.asarray([i for i in sel if i in allowed], dtype=int)
        out[tid] = sel
    return out


def load_spot_dict(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as fh:
        sd = json.load(fh)
    return {str(k): [str(c) for c in v] for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Spot-level proportions from single-nucleus predictions
# ---------------------------------------------------------------------------
def clump_to_spots(
    pred: pd.DataFrame,
    coords: pd.DataFrame,
    radius: float,
    spot_dict: Optional[Dict[str, List[str]]] = None,
    nuclei: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Average per-nucleus probabilities over the nuclei of each spot.

    This is the paper's own way of turning single-nucleus predictions back into
    spot proportions: "the predicted single-nucleus-level cells in each spot
    were clumped to mimic spot-level cell proportions".  Membership comes from
    ``spot_dict`` when supplied, otherwise from the spot disc of ``radius``.
    """
    rows, index = [], []
    if spot_dict is not None:
        for spot in coords.index:
            members = [c for c in spot_dict.get(spot, ()) if c in pred.index]
            if not members:
                continue
            rows.append(pred.loc[members].to_numpy().mean(axis=0))
            index.append(spot)
    else:
        if nuclei is None:
            raise ValueError("clump_to_spots needs `nuclei` when no spot_dict is given")
        from scipy.spatial import cKDTree

        keep = nuclei.index.intersection(pred.index)
        sub = nuclei.loc[keep]
        tree = cKDTree(sub[["cx", "cy"]].to_numpy())
        values = pred.loc[sub.index].to_numpy()
        for spot, (x, y) in zip(coords.index, coords[["x", "y"]].to_numpy()):
            idx = tree.query_ball_point([x, y], radius)
            if not idx:
                continue
            rows.append(values[idx].mean(axis=0))
            index.append(spot)

    df = pd.DataFrame(rows, index=pd.Index(index, name="spot_id"), columns=pred.columns)
    logger.info("Clumped predictions into %d/%d spots", len(df), len(coords))
    return df


# ---------------------------------------------------------------------------
# Cell-type colours
# ---------------------------------------------------------------------------
# The benchmark draws its own figures with a hand-written palette: every cell
# type has a colour chosen by hand and keyed by *name*, so a type keeps its
# colour across annotation levels, across methods and across every figure.
# Hue says which compartment a type belongs to -- lymphoid green to teal,
# myeloid light golds, epithelium dark warm reds, all of structural one blue
# ladder (stroma its dark half, vessels its light half), melanocyte near-black
# to rose magenta -- and lightness says which member, because lightness is the channel that survives
# colour blindness: under red-green blindness green, gold and red collapse onto
# one yellowish axis, so those families are kept in separate L* bands instead.
#
# Neither H&E stain colour is used.  The figures draw cells over a faded H&E
# background, so a cell coloured like eosin pink or haematoxylin purple is
# nearly invisible; the stain's own blue region is used only at
# lightnesses it does not occupy, and the proliferating melanocyte is a vivid
# rose magenta rather than the pale lilac it once was.
#
# The table is reproduced here rather than imported, so this package stays
# self-contained.  It must match ``hedest_bench.celltype_colors`` and
# ``simulations/semi_simulations/STHELAR/palette.py`` exactly;
# ``python -m hedest_bench.check_palette_sync`` in HEDeST-bench asserts it.
OTHER_COLOR = (0.63, 0.63, 0.63)
OTHER_LABEL = "Other"

CELLTYPE_HEX: Dict[str, str] = {
    # --- immune: lymphoid, green -> teal
    "Immune": "#1E8B5A",
    "Lymphoid": "#17805C",
    "B_Plasma": "#1B8A4E",
    "B-lineage": "#1B8A4E",
    "B": "#177A46",
    "Plasma": "#4FC2A1",
    "T_NK": "#0D8A86",
    "T/NK-lineage": "#0D8A86",
    "T": "#0F8F88",
    "T_CD8": "#0C7D7C",
    # --- immune: myeloid, light golds
    "Myeloid": "#D49A2E",
    "Monocyte/Macrophage": "#EEBB63",
    "Monocyte/Macrophage/DC": "#E3A944",
    "Mast": "#F7DDA0",
    "DC": "#FCF0D8",
    "Dendritic Cells": "#FCF0D8",
    # --- structural: stroma, dark half of the blue ladder
    "Structural": "#16216A",
    "Stromal": "#1A277F",
    "Fibroblast_Myofibroblast": "#2539B6",
    "CAF": "#3C51D7",
    "Fibroblast": "#7886E3",
    "Fibroblast/Myofibroblast": "#A2ACEC",
    "Myofibroblast": "#101950",
    # --- structural: vessels, the light half of the shared blue ladder
    "Blood_vessel": "#1385BE",
    "Blood vessel": "#1385BE",
    "Blood Vessel": "#1385BE",
    "Endothelial": "#1CA4E9",
    "Angiogenic_endothelial": "#99D6F5",
    "Pericyte": "#53BAEE",
    "Smooth_muscle": "#09415D",
    # --- epithelium: dark warm reds
    "Epithelial": "#A33A2B",
    "Breast_epithelial_cell": "#A33A2B",
    "Keratinocyte": "#BB4B30",
    "Luminal_epithelial_cell": "#C96A45",
    "Secretory_epithelial_cell": "#C25C3C",
    "Squamous_epithelial_cell": "#78242E",
    "Proliferating_epithelial": "#D4886B",
    "Lung_epithelial_cell": "#A33A2B",
    "Lung_alveolar_typeI_epithelial_cell": "#C26A45",
    "Lung_alveolar_typeII_epithelial_cell": "#78242E",
    "Lung_ciliated_epithelial_cell": "#D9A086",
    # --- melanocyte: near-black plum -> rose magenta
    "Melanocyte": "#2E1424",
    "High_proliferating_melanocyte": "#EB006D",
}


def celltype_rgb(name: str) -> Tuple[float, float, float]:
    """``(r, g, b)`` in 0..1 for a cell type; grey for anything not in the table."""
    value = CELLTYPE_HEX.get(name)
    if value is None:
        return OTHER_COLOR
    return tuple(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))


def level_palette(sim_dir: str, level: int) -> Dict[str, Tuple[float, float, float]]:
    """``category -> RGB (0..1)`` for one level of a slide's ``sim/`` directory.

    The categories are the columns of that level's ``proportions.csv``; each one
    is looked up by name, so no hierarchy has to be derived to colour a plot.
    Anything the table does not define is left out, which is what lets the
    caller notice and fall back rather than silently drawing it grey.
    """
    path = os.path.join(sim_dir, f"level{level}", "proportions.csv")
    cats = list(pd.read_csv(path, index_col=0, nrows=0).columns)

    return {c: celltype_rgb(c) for c in cats if c in CELLTYPE_HEX}
