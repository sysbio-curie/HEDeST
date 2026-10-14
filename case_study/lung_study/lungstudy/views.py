"""
Slide views: the tissue with the spots, the deconvolution pies and the HEDeST cells on top.

Everything is drawn in level-0 slide pixels: the tissue is shown with ``imshow(extent=...)``,
so spots, pies, contours and dots all share one coordinate system, whatever resolution the
tissue was read at.
"""
from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lungstudy.data import SampleData
from lungstudy.data import Window
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.ndimage import gaussian_filter

GREY = (0.6, 0.6, 0.6)


# ------------------------------------------------------------------ primitives


def show_tissue(ax: plt.Axes, data: SampleData, window: Window, max_pixels: int = 2000, alpha: float = 1.0):
    """Draws the tissue of a window in level-0 coordinates and returns the region read."""

    region = data.slide.read(window, max_pixels=max_pixels)
    x0, y0 = region.origin
    w, h = region.size
    ax.imshow(region.image, extent=(x0, x0 + w, y0 + h, y0), alpha=alpha, interpolation="lanczos")
    ax.set_xlim(x0, x0 + w)
    ax.set_ylim(y0 + h, y0)
    ax.set_aspect("equal")
    ax.axis("off")

    return region


def blank(ax: plt.Axes, window: Window, color: str = "white") -> None:
    """Sets up an empty panel for a window, without tissue."""

    (x0, y0), (w, h) = window
    ax.set_facecolor(color)
    ax.set_xlim(x0, x0 + w)
    ax.set_ylim(y0 + h, y0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def in_window(xy: np.ndarray, window: Window, margin: float = 0.0) -> np.ndarray:
    (x0, y0), (w, h) = window
    return (
        (xy[:, 0] >= x0 - margin)
        & (xy[:, 0] <= x0 + w + margin)
        & (xy[:, 1] >= y0 - margin)
        & (xy[:, 1] <= y0 + h + margin)
    )


def add_spot_circles(
    ax: plt.Axes,
    data: SampleData,
    window: Window,
    color="black",
    linewidth: float = 1.0,
    spots: Optional[Iterable[str]] = None,
) -> None:
    """Draws the spots as circles of their real 55 µm diameter."""

    table = data.spots if spots is None else data.spots.loc[list(spots)]
    xy = table[["x", "y"]].to_numpy()
    keep = in_window(xy, window, data.radius_px)
    patches = [Circle((x, y), data.radius_px) for x, y in xy[keep]]
    ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor=color, linewidth=linewidth))


def add_pies(
    ax: plt.Axes,
    data: SampleData,
    window: Window,
    props: Optional[pd.DataFrame] = None,
    edgecolor="white",
    linewidth: float = 0.3,
    alpha: float = 1.0,
    radius_scale: float = 1.0,
) -> None:
    """Draws every spot of a window as a pie of its proportions (the deconvolution by default)."""

    props = data.props if props is None else props
    table = data.spots.loc[data.spots.index.intersection(props.index)]
    xy = table[["x", "y"]].to_numpy()
    keep = in_window(xy, window, data.radius_px)
    radius = data.radius_px * radius_scale
    patches, colors = [], []
    for spot, (x, y) in zip(table.index[keep], xy[keep]):
        row = props.loc[spot]
        row = row[row > 0]
        start = 90.0
        for cell_type, value in row.items():
            angle = 360.0 * float(value) / float(row.sum())
            patches.append(Wedge((x, y), radius, start - angle, start))
            colors.append(data.palette.rgb(cell_type) if cell_type in data.palette else GREY)
            start -= angle
    collection = PatchCollection(patches, facecolor=colors, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
    ax.add_collection(collection)


def add_cells(
    ax: plt.Axes,
    data: SampleData,
    window: Window,
    labels: Optional[pd.Series] = None,
    contours: bool = True,
    alpha: float = 0.55,
    edge_alpha: float = 1.0,
    linewidth: float = 0.6,
    size: Optional[float] = None,
    cell_types: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, tuple]] = None,
    other_color=None,
) -> int:
    """
    Draws the HEDeST cells of a window, as filled contours (zooms) or dots (large windows).

    Args:
        labels: Cell type per cell id. HEDeST's by default.
        cell_types: Only these types are drawn in colour; the others in ``other_color``
            (skipped when None).
        colors: Overrides the palette for some labels.

    Returns:
        The number of cells drawn.
    """

    labels = data.cells["label"] if labels is None else labels
    xy = data.cells[["x", "y"]].to_numpy()
    keep = in_window(xy, window, 30)
    ids = data.cells.index[keep]
    names = labels.reindex(ids).to_numpy()

    def color_of(name):
        if colors and name in colors:
            return colors[name]
        if cell_types is not None and name not in cell_types:
            return other_color
        if name in data.palette:
            return data.palette.rgb(name)
        return GREY

    rgb = [color_of(n) for n in names]
    drawn = np.array([c is not None for c in rgb], dtype=bool)
    ids, rgb = ids[drawn], [c for c in rgb if c is not None]
    if not len(ids):
        return 0

    if contours:
        polygons = [data.contours[c] for c in ids]
        face = [(*c, alpha) for c in rgb]
        edge = [(*c, edge_alpha) for c in rgb]
        ax.add_collection(PolyCollection(polygons, facecolors=face, edgecolors=edge, linewidths=linewidth))
    else:
        points = data.cells.loc[ids, ["x", "y"]].to_numpy()
        if size is None:
            size = nucleus_marker_size(ax, data, window)
        ax.scatter(points[:, 0], points[:, 1], s=size, c=rgb, alpha=alpha, linewidths=0, rasterized=True)

    return len(ids)


def nucleus_marker_size(ax: plt.Axes, data: SampleData, window: Window, diameter_um: float = 9.0) -> float:
    """The scatter marker area (pt²) of a dot as wide as a nucleus, at least visible."""

    width_points = ax.get_window_extent().width * 72.0 / ax.figure.dpi
    points_per_px = width_points / window[1][0]
    diameter_points = diameter_um / data.mpp * points_per_px

    return float(np.clip(diameter_points**2, 0.25, 40.0))


def add_scalebar(ax: plt.Axes, data: SampleData, length_um: float, color="black", loc="lower right") -> None:
    bar = AnchoredSizeBar(
        ax.transData,
        length_um / data.mpp,
        f"{length_um:g} µm",
        loc,
        pad=0.4,
        color=color,
        frameon=False,
        size_vertical=length_um / data.mpp / 40,
        fontproperties={"size": 9},
    )
    ax.add_artist(bar)


def add_legend(
    target,
    data: SampleData,
    names: Optional[Sequence[str]] = None,
    ncol: int = 1,
    loc="center left",
    anchor=(1.01, 0.5),
    fontsize=9,
    title=None,
) -> None:
    names = list(names) if names is not None else data.ct
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=data.palette.rgb(n), markersize=9, label=n)
        for n in names
        if n in data.palette
    ]
    target.legend(
        handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=ncol, frameon=False, fontsize=fontsize, title=title
    )


def window_box(ax: plt.Axes, window: Window, color="black", linewidth=1.5, label: Optional[str] = None) -> None:
    (x0, y0), (w, h) = window
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=linewidth))
    if label:
        ax.text(x0, y0, label, color=color, fontsize=10, ha="left", va="bottom", fontweight="bold")


# ------------------------------------------------------------------ composite views


def compare_panels(
    data: SampleData,
    window: Window,
    panels: Sequence[str] = ("tissue", "pies", "cells"),
    panel_size: float = 5.5,
    contours: Optional[bool] = None,
    scalebar_um: Optional[float] = None,
    title: Optional[str] = None,
    legend: bool = True,
    max_pixels: int = 2000,
    cell_types: Optional[Sequence[str]] = None,
):
    """
    The same window seen several ways, side by side:

    - ``tissue``: H&E with the spots as circles;
    - ``pies``: H&E with the deconvolution as a pie per spot;
    - ``cells``: H&E with the HEDeST cells (filled contours), spots as circles;
    - ``cells_only``: the HEDeST cells on white, spots as circles;
    - ``hedest_pies``: H&E with the HEDeST composition of each spot as a pie.
    """

    size_um = window[1][0] * data.mpp
    if contours is None:
        contours = size_um <= 700
    if scalebar_um is None:
        scalebar_um = 50 if size_um <= 400 else (100 if size_um <= 900 else (500 if size_um <= 4000 else 1000))

    titles = {
        "tissue": "H&E + Visium spots (55 µm)",
        "pies": "deconvolution (one pie per spot)",
        "cells": "HEDeST (one type per cell)",
        "cells_only": "HEDeST cells",
        "hedest_pies": "HEDeST cells counted per spot",
    }
    fig, axes = plt.subplots(1, len(panels), figsize=(panel_size * len(panels), panel_size))
    axes = np.atleast_1d(axes)
    shown_types = set()
    for ax, kind in zip(axes, panels):
        if kind == "cells_only":
            blank(ax, window)
        else:
            show_tissue(ax, data, window, max_pixels=max_pixels, alpha=0.55 if kind == "hedest_pies" else 1.0)
        if kind == "tissue":
            add_spot_circles(ax, data, window, color="black", linewidth=1.0)
        elif kind == "pies":
            add_pies(ax, data, window)
        elif kind == "hedest_pies":
            add_pies(ax, data, window, props=data.hedest_spot_composition())
        elif kind in ("cells", "cells_only"):
            add_cells(ax, data, window, contours=contours, alpha=0.6 if contours else 0.9, cell_types=cell_types)
            add_spot_circles(ax, data, window, color="black", linewidth=0.8)
            xy = data.cells[["x", "y"]].to_numpy()
            shown_types |= set(data.cells.loc[in_window(xy, window), "label"])
        add_scalebar(ax, data, scalebar_um)
        ax.set_title(titles.get(kind, kind), fontsize=11)
    if legend:
        props_types = set()
        if any(k in panels for k in ("pies", "hedest_pies")):
            xy = data.spots.loc[data.props.index, ["x", "y"]].to_numpy()
            local = data.props.loc[in_window(xy, window, data.radius_px)]
            props_types = set(local.columns[(local > 0).any(axis=0)])
        names = [c for c in data.ct if c in (shown_types | props_types)]
        add_legend(axes[-1], data, names)
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    return fig


def whole_slide_compare(
    data: SampleData, max_pixels: int = 3000, panel_size: float = 8.0, point_size: Optional[float] = None
):
    """The whole slide: H&E, deconvolution pies, HEDeST cells."""

    full = ((0, 0), tuple(float(v) for v in data.slide.dimensions))
    xy = data.cells[["x", "y"]].to_numpy()
    pad = 3 * data.pitch_px
    x0, y0 = xy.min(axis=0) - pad
    x1, y1 = xy.max(axis=0) + pad
    window = ((max(0, x0), max(0, y0)), (min(full[1][0], x1) - max(0, x0), min(full[1][1], y1) - max(0, y0)))

    fig, axes = plt.subplots(1, 3, figsize=(3 * panel_size, panel_size * window[1][1] / window[1][0] + 0.6))
    show_tissue(axes[0], data, window, max_pixels=max_pixels)
    axes[0].set_title("H&E", fontsize=12)
    show_tissue(axes[1], data, window, max_pixels=max_pixels, alpha=0.35)
    add_pies(axes[1], data, window, linewidth=0)
    axes[1].set_title(f"deconvolution: {len(data.props):,} spots", fontsize=12)
    show_tissue(axes[2], data, window, max_pixels=max_pixels, alpha=0.35)
    fig.tight_layout()
    add_cells(axes[2], data, window, contours=False, size=point_size, alpha=1.0)
    axes[2].set_title(f"HEDeST: {len(data.cells):,} cells", fontsize=12)
    for ax in axes:
        add_scalebar(ax, data, 1000)
    present = set(data.cells["label"]) | set(data.props.columns[(data.props > 0).any(axis=0)])
    add_legend(axes[2], data, [c for c in data.ct if c in present])

    return fig, window


def slide_extent(data: SampleData, pad_pitch: float = 3.0) -> Window:
    """The window holding every cell of the slide, with a margin."""

    xy = data.cells[["x", "y"]].to_numpy()
    pad = pad_pitch * data.pitch_px
    x0, y0 = np.maximum(xy.min(axis=0) - pad, 0)
    x1, y1 = xy.max(axis=0) + pad
    width, height = data.slide.dimensions
    return ((x0, y0), (min(width, x1) - x0, min(height, y1) - y0))


def type_maps(
    data: SampleData,
    cell_types: Sequence[str],
    bin_um: float = 20.0,
    sigma_um: float = 15.0,
    window: Optional[Window] = None,
    panel_size: float = 5.0,
    cmap: str = "magma_r",
    vmax: Optional[float] = None,
):
    """
    For each cell type, two maps of the same quantity, the share of the type:

    - top: per spot, its deconvolution proportion (one disc per spot);
    - bottom: per 20 µm pixel, the share of the type among the HEDeST cells around (Gaussian
      of 15 µm), wherever there are cells, between the spots included.
    """

    window = window or slide_extent(data)
    (x0, y0), (w, h) = window
    bin_px = bin_um / data.mpp
    nx, ny = int(np.ceil(w / bin_px)), int(np.ceil(h / bin_px))
    xy = data.cells[["x", "y"]].to_numpy()
    keep = in_window(xy, window)
    gx = np.clip(((xy[keep, 0] - x0) / bin_px).astype(int), 0, nx - 1)
    gy = np.clip(((xy[keep, 1] - y0) / bin_px).astype(int), 0, ny - 1)
    labels = data.cells["label"].to_numpy()[keep]
    total = np.zeros((ny, nx))
    np.add.at(total, (gy, gx), 1)
    total_s = gaussian_filter(total, sigma_um / bin_um)
    tissue = gaussian_filter((total > 0).astype(float), 1.0) > 0.2

    fig, axes = plt.subplots(
        2, len(cell_types), figsize=(panel_size * len(cell_types), 2 * panel_size * h / w + 1.6), layout="constrained"
    )
    axes = np.atleast_2d(axes).reshape(2, len(cell_types))
    spots_xy = data.spots.loc[data.props.index, ["x", "y"]].to_numpy()
    for column, cell_type in enumerate(cell_types):
        top = vmax or float(np.quantile(data.props[cell_type], 0.99)) or 1.0
        ax = axes[0, column]
        blank(ax, window, color="#f2f2f2")
        values = data.props[cell_type].to_numpy()
        patches = [Circle((x, y), data.radius_px) for x, y in spots_xy]
        collection = PatchCollection(patches, cmap=cmap, edgecolor="none")
        collection.set_array(values)
        collection.set_clim(0, top)
        ax.add_collection(collection)
        ax.set_title(f"{cell_type}\ndeconvolution, per spot", fontsize=11)
        add_scalebar(ax, data, 1000)

        count = np.zeros((ny, nx))
        mask = labels == cell_type
        np.add.at(count, (gy[mask], gx[mask]), 1)
        share = gaussian_filter(count, sigma_um / bin_um) / np.maximum(total_s, 1e-9)
        share[~tissue] = np.nan
        ax = axes[1, column]
        blank(ax, window, color="#f2f2f2")
        image = ax.imshow(
            share,
            extent=(x0, x0 + nx * bin_px, y0 + ny * bin_px, y0),
            cmap=cmap,
            vmin=0,
            vmax=top,
            interpolation="nearest",
        )
        ax.set_xlim(x0, x0 + w)
        ax.set_ylim(y0 + h, y0)
        ax.set_title(f"HEDeST, per {bin_um:g} µm pixel", fontsize=11)
        add_scalebar(ax, data, 1000)
        fig.colorbar(
            image, ax=axes[1, column], orientation="horizontal", fraction=0.05, pad=0.02, label=f"share of {cell_type}"
        )

    return fig


def zoom_series(
    data: SampleData,
    x: float,
    y: float,
    sizes_um: Sequence[float] = (2000, 600, 250),
    max_pixels: int = 2000,
    panel_size: float = 6.0,
):
    """Zooms of increasing magnification on one point, HEDeST cells over H&E, each showing the next window."""

    fig, axes = plt.subplots(1, len(sizes_um), figsize=(panel_size * len(sizes_um), panel_size))
    windows = [data.window(x, y, s) for s in sizes_um]
    for index, (ax, window, size) in enumerate(zip(axes, windows, sizes_um)):
        show_tissue(ax, data, window, max_pixels=max_pixels)
        add_cells(ax, data, window, contours=size <= 700, alpha=0.6 if size <= 700 else 0.9)
        add_spot_circles(ax, data, window, color="black", linewidth=0.6 if size > 700 else 1.0)
        if index + 1 < len(windows):
            window_box(ax, windows[index + 1], color="black", linewidth=1.5)
        add_scalebar(ax, data, 500 if size > 1500 else (100 if size > 400 else 50))
        ax.set_title(f"{size:g} µm", fontsize=11)
    xy = data.cells[["x", "y"]].to_numpy()
    shown = set(data.cells.loc[in_window(xy, windows[0]), "label"])
    add_legend(axes[-1], data, [c for c in data.ct if c in shown])
    fig.tight_layout()

    return fig


def transect_profile(
    data: SampleData,
    start: Sequence[float],
    end: Sequence[float],
    types: Sequence[str],
    step_um: float = 5.0,
    radius_um: float = 15.0,
) -> pd.DataFrame:
    """
    The share of each type along a line (level-0 pixels from ``start`` to ``end``):

    - ``hedest_*``: among the HEDeST cells within ``radius_um`` of each point;
    - ``deconv_*``: the deconvolution of the spot the point falls in (NaN between spots);
    - ``nearest_*``: the deconvolution of the nearest spot (a step function).
    """

    from scipy.spatial import cKDTree

    start, end = np.asarray(start, float), np.asarray(end, float)
    length_px = np.linalg.norm(end - start)
    steps = int(length_px * data.mpp / step_um) + 1
    points = start + np.linspace(0, 1, steps)[:, None] * (end - start)
    tree = cKDTree(data.cells[["x", "y"]].to_numpy())
    labels = data.cells["label"].to_numpy()
    neighbours = tree.query_ball_point(points, r=radius_um / data.mpp)
    spot_xy = data.spots.loc[data.props.index, ["x", "y"]].to_numpy()
    distance, nearest = cKDTree(spot_xy).query(points)
    rows = []
    for i, point in enumerate(points):
        row = {"position_um": i * step_um, "n_cells": len(neighbours[i])}
        names = labels[neighbours[i]]
        spot = data.props.index[nearest[i]]
        for cell_type in types:
            row[f"hedest_{cell_type}"] = float((names == cell_type).mean()) if len(names) else np.nan
            value = float(data.props.loc[spot, cell_type])
            row[f"nearest_{cell_type}"] = value
            row[f"deconv_{cell_type}"] = value if distance[i] <= data.radius_px else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def per_type_panels(
    data: SampleData, window: Window, ncols: int = 4, panel_size: float = 4.0, contours: Optional[bool] = None
):
    """One panel per cell type: that type in colour, every other cell grey, spots as circles."""

    size_um = window[1][0] * data.mpp
    contours = size_um <= 700 if contours is None else contours
    xy = data.cells[["x", "y"]].to_numpy()
    present = data.cells.loc[in_window(xy, window), "label"].value_counts()
    types = [c for c in data.ct if present.get(c, 0) > 0]
    nrows = int(np.ceil(len(types) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, cell_type in zip(np.ravel(axes), types):
        show_tissue(ax, data, window, alpha=0.45)
        add_cells(
            ax, data, window, contours=contours, cell_types=[cell_type], other_color=(0.75, 0.75, 0.75), alpha=0.8
        )
        add_spot_circles(ax, data, window, color="black", linewidth=0.5)
        ax.set_title(f"{cell_type} ({present.get(cell_type, 0):,})", fontsize=11, color=data.palette.rgb(cell_type))
    fig.tight_layout()

    return fig


def gallery_rows(
    image_dict: dict,
    rows: Dict[str, List[str]],
    colors: Dict[str, tuple],
    ncols: int = 12,
    cell_size: float = 0.7,
    title: Optional[str] = None,
):
    """A gallery with one row of crops per named group of cells."""

    from hedest.analysis.plots import as_image

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_size + 2.2, nrows * cell_size + 0.4))
    axes = np.atleast_2d(axes)
    for r, (name, ids) in enumerate(rows.items()):
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(colors.get(name, "black"))
                spine.set_linewidth(2)
            if c < len(ids) and ids[c] in image_dict:
                ax.imshow(as_image(image_dict[ids[c]]))
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9, color=colors.get(name, "black"))
    if title:
        fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.2)

    return fig
