"""
Everything about one sample, loaded from the cache, at one level of annotation.

    data = SampleData("A_AAAA1879")          # lv3, 10pct_filter, HEDeST without PPSA
    data.cells        # one row per nucleus, with its HEDeST label, probability, seed agreement
    data.pred         # cells x cell types, mean of the 5 seeds
    data.props        # spots x cell types, the deconvolution HEDeST was trained on
    data.spots        # spot centres
"""
from __future__ import annotations

from functools import cached_property
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from lungstudy import cache
from lungstudy import paths
from scipy.spatial import cKDTree

from hedest.analysis.palette import Palette

Window = Tuple[Tuple[float, float], Tuple[float, float]]


class SampleData:
    """The cached tables of one sample at one level, plus its slide and contours on demand."""

    def __init__(self, name: str, level: str = paths.MAIN_LEVEL, kind: str = paths.PROPORTIONS) -> None:
        self.name = name
        self.level = level
        self.kind = kind
        self.meta = cache.meta(name)
        self.mpp = float(self.meta["mpp"])
        self.radius_px = self.meta["spot_diameter_px"] / 2.0
        self.pitch_px = float(self.meta["spot_pitch_px"])

        self.cells = cache.table(name, level, kind)
        self.pred = cache.predictions(name, level, kind)
        self.ct = list(self.pred.columns)
        self.palette = Palette(self.ct)

        spots = cache.spots(name)
        props = cache.proportions(name, level, kind)
        props = props.loc[props.index.intersection(spots.index)].dropna(how="any")
        self.props = props[self.ct]
        self.spots = spots

    def __repr__(self) -> str:
        return (
            f"SampleData({self.name}, {self.level}/{self.kind}, {len(self.cells):,} cells, "
            f"{len(self.props):,} spots with proportions)"
        )

    # ------------------------------------------------------------------ lazily loaded

    @cached_property
    def contours(self) -> Dict[str, np.ndarray]:
        return cache.contours(self.name)

    @cached_property
    def slide(self):
        from hedest.analysis.postseg import SlideVisualizer

        return SlideVisualizer(self.meta["slide"], mpp=self.mpp)

    def seg_dict(self) -> dict:
        """The segmentation in the HoVer-Net layout HEDeST's analysis tools take, without reading the JSON."""

        xy = self.cells[["x", "y"]].to_numpy()
        types = self.cells["hv_type"].to_numpy()
        return {
            "nuc": {
                cell: {"centroid": [float(x), float(y)], "contour": self.contours[cell], "type": int(t)}
                for cell, (x, y), t in zip(self.cells.index, xy, types)
            }
        }

    @cached_property
    def spot_tree(self) -> cKDTree:
        return cKDTree(self.spots[["x", "y"]].to_numpy())

    # ------------------------------------------------------------------ helpers

    def window(self, x: float, y: float, size_um: float) -> Window:
        """A square window of ``size_um`` microns centred on (x, y), in level-0 pixels."""

        side = size_um / self.mpp
        return ((x - side / 2, y - side / 2), (side, side))

    def spot_xy(self, spot: str) -> Tuple[float, float]:
        row = self.spots.loc[spot]
        return float(row["x"]), float(row["y"])

    def hedest_spot_composition(self, min_cells: int = 1) -> pd.DataFrame:
        """The share of each HEDeST cell type among the cells of each spot."""

        inside = self.cells[self.cells["in_spot"]]
        table = pd.crosstab(inside["spot"], inside["label"]).reindex(columns=self.ct, fill_value=0)
        table = table[table.sum(axis=1) >= min_cells]

        return table.div(table.sum(axis=1), axis=0)

    def spot_neighbours(self, radius_pitch: float = 1.2) -> Dict[str, list]:
        """The spots within ``radius_pitch`` pitches of each spot (itself excluded)."""

        xy = self.spots[["x", "y"]].to_numpy()
        ids = np.asarray(self.spots.index)
        lists = self.spot_tree.query_ball_point(xy, r=radius_pitch * self.pitch_px)

        return {ids[i]: [ids[j] for j in members if j != i] for i, members in enumerate(lists)}

    def smoothed_props(self, radius_pitch: float = 1.2) -> pd.DataFrame:
        """The proportions averaged over each spot and its first ring of neighbours."""

        neighbours = self.spot_neighbours(radius_pitch)
        rows = {}
        for spot in self.props.index:
            members = [spot] + [s for s in neighbours.get(spot, []) if s in self.props.index]
            rows[spot] = self.props.loc[members].mean(axis=0)

        return pd.DataFrame(rows).T[self.ct]


def lymphoid_types(ct: list) -> list:
    """The lymphocyte classes of a level, without NK cells (whose proportions track the epithelium here)."""

    wanted = {"B cell", "T cell", "T CD4", "T CD8", "plasma cell", "immune_lymphoid", "T helper", "T reg"}
    return [c for c in ct if c in wanted or c.startswith("T CD8")]


def pick_sites(data: SampleData, min_cells: int = 8) -> Dict[str, dict]:
    """
    Picks the places of a slide worth zooming on, from the deconvolution alone.

    Returns:
        name -> {"spot", "x", "y", "why"}.
    """

    props = data.props
    smooth = data.smoothed_props()
    counts = data.spots["n_cells"].reindex(props.index).fillna(0)
    valid = counts >= min_cells

    def best(score: pd.Series) -> str:
        return str(score[valid.reindex(score.index).fillna(False)].idxmax())

    lymph = [c for c in ("B cell", "T cell") if c in smooth]
    vascular = [c for c in ("endothelial", "SMC", "perictyes") if c in smooth]
    sites = {}
    if lymph:
        sites["lymphoid"] = (best(smooth[lymph].sum(axis=1)), "highest B + T proportion (spot and its neighbours)")
    if "epithelial" in props and "CAF/fibro" in props:
        sites["interface"] = (
            best(props["epithelial"] * props["CAF/fibro"]),
            "epithelial and CAF/fibro both abundant in one spot",
        )
    if "epithelial" in smooth:
        sites["epithelial"] = (best(smooth["epithelial"]), "highest epithelial proportion (spot and its neighbours)")
    if vascular:
        sites["vascular"] = (best(smooth[vascular].sum(axis=1)), "highest endothelial + SMC + pericyte proportion")
    if "macrophage" in smooth:
        sites["myeloid"] = (best(smooth["macrophage"]), "highest macrophage proportion (spot and its neighbours)")
    sites["busiest"] = (str(counts.idxmax()), "spot holding the most nuclei")

    out = {}
    for name, (spot, why) in sites.items():
        x, y = data.spot_xy(spot)
        out[name] = {"spot": spot, "x": x, "y": y, "why": why}

    return out


def other_level(name: str, level: str, kind: Optional[str] = None) -> SampleData:
    """Shortcut for another level of the same sample."""

    return SampleData(name, level, kind or paths.PROPORTIONS)
