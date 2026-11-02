"""
Figures that are not slide views: the colocalization zone plot, readable labels, and the
H&E crops that show a pair of cell types.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lungstudy import coloc as C
from lungstudy import views
from lungstudy.data import SampleData
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

ZONE_ORDER = ["same", "more", "flip_together", "less", "flip_apart"]
ZONE_COLORS = {
    "same": "#e9e9e9",
    "more": "#cfe3f5",
    "flip_together": "#7fb2e0",
    "less": "#fbe0c4",
    "flip_apart": "#f2a36b",
}
POINT_COLORS = {
    "same": "#8c8c8c",
    "more": "#1f5fa6",
    "flip_together": "#1f5fa6",
    "less": "#b35806",
    "flip_apart": "#b35806",
}
IMMUNE = {"B cell", "T cell", "macrophage"}


def place_labels(
    ax: plt.Axes,
    points: np.ndarray,
    texts: Sequence[str],
    colors: Sequence,
    avoid: Optional[np.ndarray] = None,
    fontsize: float = 11,
) -> List[str]:
    """
    Labels points with leader lines, trying positions at growing distances until the label
    overlaps neither another label nor a point. Returns the labels it could not place.
    """

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    obstacles = ax.transData.transform(points if avoid is None else avoid)
    candidates = [
        (radius * np.cos(angle), radius * np.sin(angle))
        for radius in (12, 28, 46, 70, 100, 140)
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ]
    taken, missed = [], []
    for (x, y), text, color in zip(points, texts, colors):
        for dx, dy in candidates:
            label = ax.annotate(
                text,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                color=color,
                fontweight="bold",
                ha="left" if dx > 3 else ("right" if dx < -3 else "center"),
                va="bottom" if dy > 3 else ("top" if dy < -3 else "center"),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.8, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.9, shrinkA=0, shrinkB=4),
                zorder=6,
            )
            box = label.get_window_extent(renderer).expanded(1.03, 1.08)
            fits = frame.contains(box.x0, box.y0) and frame.contains(box.x1, box.y1)
            if fits and not any(box.overlaps(o) for o in taken) and not any(box.contains(*q) for q in obstacles):
                taken.append(box)
                break
            label.remove()
        else:
            missed.append(text)

    return missed


def zone_background(ax: plt.Axes, low: float, high: float, band: float = C.BAND) -> None:
    """Shades the five zones of :data:`lungstudy.coloc.ZONES` and writes their names."""

    grid = np.linspace(low, high, 800)
    x, y = np.meshgrid(grid, grid)
    gap = y - x
    codes = np.zeros_like(x, dtype=int)
    codes[gap >= band] = 1
    codes[gap <= -band] = 3
    codes[(gap >= band) & (x < 0) & (y > 0)] = 2
    codes[(gap <= -band) & (x > 0) & (y < 0)] = 4
    ax.imshow(
        codes,
        extent=(low, high, low, high),
        origin="lower",
        cmap=ListedColormap([ZONE_COLORS[z] for z in ZONE_ORDER]),
        vmin=-0.5,
        vmax=4.5,
        interpolation="nearest",
        zorder=0,
        aspect="auto",
    )
    ax.axhline(0, color="#555555", lw=0.8, zorder=1)
    ax.axvline(0, color="#555555", lw=0.8, zorder=1)
    ax.plot([low, high], [low, high], color="#777777", lw=0.8, ls="--", zorder=1)

    span = high - low
    style = dict(fontsize=12, ha="center", va="center", zorder=4, style="italic", fontweight="bold")
    ax.text(low + 0.2 * span, high - 0.06 * span, "spots: apart\ncells: together", color="#16467a", **style)
    ax.text(high - 0.2 * span, low + 0.06 * span, "spots: together\ncells: apart", color="#7a3a05", **style)
    ax.text(low + 0.1 * span, low + 0.45 * span, "HEDeST:\nmore\ncoloc.", color="#1f5fa6", **style)
    ax.text(low + 0.38 * span, low + 0.04 * span, "HEDeST: less colocalization", color="#b35806", **style)
    ax.text(high - 0.12 * span, high - 0.2 * span, "same\nanswer", color="#555555", **style)


def zone_panel(ax: plt.Axes, pairs: pd.DataFrame, title: str, max_labels: int = 9) -> pd.DataFrame:
    """
    Every pair of one slide: deconvolution co-occurrence (x) against HEDeST contact (y), with
    95% intervals, over the zones. Pairs off the 'same answer' band whose interval excludes
    no difference are filled; the most telling of them are named (pairs with an immune type
    first). Returns the labelled pairs.
    """

    values = np.r_[pairs["deconv_cooc"], pairs["contact"]]
    low = max(np.floor(values.min() - 0.7), -5.5)
    high = min(np.ceil(values.max() + 0.7), 5.5)
    zone_background(ax, low, high)
    for _, row in pairs.iterrows():
        x, y = row["deconv_cooc"], row["contact"]
        color = POINT_COLORS[row["zone"]] if row["robust"] else "#8c8c8c"
        ax.errorbar(
            x,
            y,
            xerr=[[x - row["deconv_cooc_low"]], [row["deconv_cooc_high"] - x]],
            yerr=[[y - row["contact_low"]], [row["contact_high"] - y]],
            fmt="none",
            ecolor=color,
            alpha=0.55,
            elinewidth=0.9,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            marker="D" if row["zone"].startswith("flip") and row["robust"] else "o",
            s=70,
            facecolor=color if row["robust"] else "white",
            edgecolor=color if row["robust"] else "#555555",
            lw=1.3,
            zorder=3,
        )
    named = pairs[(pairs["zone"] != "same") & pairs["robust"]].copy()
    named["priority"] = (
        3 * named["zone"].str.startswith("flip")
        + 2 * (named["A"].isin(IMMUNE) | named["B"].isin(IMMUNE))
        + 2 * (named["A"].isin(IMMUNE) & named["B"].isin(IMMUNE))
        + named["gap"].abs() / 2
    )
    named = named.sort_values("priority", ascending=False).head(max_labels)
    place_labels(
        ax,
        named[["deconv_cooc", "contact"]].to_numpy(),
        list(named["pair"]),
        [POINT_COLORS[z] for z in named["zone"]],
        avoid=pairs[["deconv_cooc", "contact"]].to_numpy(),
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("deconvolution: co-occurrence in spots (log2)", fontsize=12)
    ax.set_ylabel("HEDeST: enrichment at contact, touching cells (log2)", fontsize=12)

    return named


def pair_crops(
    data: SampleData,
    a: str,
    b: str,
    windows: pd.DataFrame,
    size_um: float = 220.0,
    title: Optional[str] = None,
    max_segments: int = 250,
):
    """
    For each place (rows of :func:`lungstudy.coloc.example_windows`): the H&E, the
    deconvolution pies, and the HEDeST cells of the two types with their contacts (touching
    pairs) drawn as black segments between the nucleus centres.
    """

    fig, axes = plt.subplots(len(windows), 3, figsize=(16.5, 5.7 * len(windows)))
    axes = np.atleast_2d(axes)
    for i, (_, place) in enumerate(windows.iterrows()):
        window = data.window(place["x"], place["y"], size_um)
        views.show_tissue(axes[i, 0], data, window)
        axes[i, 0].set_title("H&E", fontsize=12)
        views.show_tissue(axes[i, 1], data, window)
        views.add_pies(axes[i, 1], data, window)
        pa, pb = data.props.loc[place["spot"], a], data.props.loc[place["spot"], b]
        axes[i, 1].set_title(f"deconvolution: nearest spot {a} {pa:.0%}, {b} {pb:.0%}", fontsize=12)
        views.show_tissue(axes[i, 2], data, window, alpha=0.75)
        views.add_cells(axes[i, 2], data, window, contours=True, cell_types=[a, b], alpha=0.9, linewidth=1.0)
        views.add_spot_circles(axes[i, 2], data, window, color="black", linewidth=0.8)
        segments = C.contact_segments(data, a, b, window)
        if 0 < len(segments) <= max_segments:
            axes[i, 2].add_collection(LineCollection(segments, colors="black", linewidths=1.4, zorder=5))
        ratio = (place["observed"] + C.PSEUDO) / (place["expected"] + C.PSEUDO)
        axes[i, 2].set_title(
            f"HEDeST: {a} and {b} only"
            + (", touching pairs in black" if len(segments) <= max_segments else "")
            + f"\n{int(place['observed'])} contacts, {place['expected']:.0f} if mixed at random (×{ratio:.2f})",
            fontsize=11,
        )
        for ax in axes[i]:
            views.add_scalebar(ax, data, 50)
    shown = set()
    for _, place in windows.iterrows():
        window = data.window(place["x"], place["y"], size_um)
        xy = data.spots.loc[data.props.index, ["x", "y"]].to_numpy()
        local = data.props.loc[views.in_window(xy, window, data.radius_px)]
        shown |= set(local.columns[(local > 0).any(axis=0)])
    views.add_legend(axes[0, 2], data, [c for c in style_order() if c in shown | {a, b}], title="pies and cells")
    if title:
        fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()

    return fig


def zone_counts(pairs: pd.DataFrame, samples: Sequence[str]) -> pd.DataFrame:
    """Number of pairs per zone and slide (pairs whose interval does not exclude 0 counted as 'same')."""

    called = np.where(pairs["robust"], pairs["zone"], "same")
    table = pd.crosstab(pairs["sample"], called).reindex(index=list(samples), columns=ZONE_ORDER, fill_value=0)

    return table


def crop_panels(
    data: SampleData,
    window,
    panels: Sequence[dict],
    title: Optional[str] = None,
    outlines: Sequence[tuple] = (),
    panel_size: float = 6.0,
    scalebar_um: Optional[float] = None,
    max_pixels: int = 2000,
    spot_circles: bool = True,
    axes=None,
):
    """
    One window seen several ways, side by side, all with the study palette (a type has the same
    colour in the pies and on the cells). Each panel is a dict:

    - ``{"kind": "he"}``: the H&E alone;
    - ``{"kind": "pies", "props": table}``: one pie per spot (default: the deconvolution);
    - ``{"kind": "cells", "labels": series, "highlight": [types]}``: the HEDeST cells; with
      ``highlight``, the other cells are drawn light grey;

    plus an optional ``"title"``. ``outlines``: (polygon in µm, colour, linestyle, linewidth)
    drawn on every panel.
    """

    from lungstudy import tls

    size_um = window[1][0] * data.mpp
    contours = size_um <= 700
    if scalebar_um is None:
        scalebar_um = 20 if size_um <= 150 else (50 if size_um <= 400 else (100 if size_um <= 900 else 250))
    if axes is None:
        fig, axes = plt.subplots(1, len(panels), figsize=(panel_size * len(panels), panel_size))
    else:
        fig = np.atleast_1d(axes)[0].figure
    axes = np.atleast_1d(axes)
    shown = []
    for ax, panel in zip(axes, panels):
        kind = panel["kind"]
        views.show_tissue(ax, data, window, max_pixels=max_pixels, alpha=0.7 if kind == "cells" else 1.0)
        if kind == "pies":
            props = panel.get("props", data.props)
            views.add_pies(ax, data, window, props=props)
            xy = data.spots.loc[data.spots.index.intersection(props.index), ["x", "y"]].to_numpy()
            local = props.loc[data.spots.index.intersection(props.index)][views.in_window(xy, window, data.radius_px)]
            shown += [c for c in local.columns[(local > 0).any(axis=0)] if c not in shown]
        elif kind == "cells":
            labels = panel.get("labels", data.cells["label"]).reindex(data.cells.index)
            highlight = panel.get("highlight")
            views.add_cells(
                ax,
                data,
                window,
                labels=labels,
                contours=contours,
                alpha=0.8 if contours else 0.95,
                cell_types=highlight,
                other_color=(0.82, 0.82, 0.82) if highlight else None,
            )
            if spot_circles:
                views.add_spot_circles(ax, data, window, color="black", linewidth=0.6)
            inside = labels[views.in_window(data.cells[["x", "y"]].to_numpy(), window)]
            present = highlight if highlight else list(inside.value_counts().index)
            shown += [c for c in present if c not in shown and c in set(inside)]
        for polygon, color, linestyle, linewidth in outlines:
            tls.draw_polygon(ax, polygon, data.mpp, color, linewidth, linestyle=linestyle)
        views.add_scalebar(ax, data, scalebar_um)
        ax.set_title(
            panel.get(
                "title",
                {"he": "H&E", "pies": "deconvolution (one pie per spot)", "cells": "HEDeST (one type per cell)"}[kind],
            ),
            fontsize=11,
        )
    order = [c for c in style_order() if c in shown] + [c for c in shown if c not in style_order()]
    views.add_legend(axes[-1], data, order)
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    return fig


def style_order() -> List[str]:
    from lungstudy.style import COLORS

    return list(COLORS)
