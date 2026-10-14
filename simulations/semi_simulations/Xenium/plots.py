"""
The figures of the Xenium semi-simulations: proportions against predictions, and galleries.

Two figures per sample, the same two for lung and for breast:

* ``{sample}_proportions_vs_predictions`` -- one row of ``1 + n`` panels. The first panel is
  the slide with the simulated spot proportions drawn as a pie per spot; the ``n`` others are
  the same slide with every predicted cell as a point, one panel per cell type, where only
  the cells of that type are coloured and the rest are left grey. Reading the row left to
  right therefore compares what the spots asked for with what HEDeST put inside them.
* ``{sample}_celltype_gallery`` -- the nucleus crops of the cells predicted for each type,
  the most confident first, through :func:`hedest.analysis.plots.plot_gallery`.

and one ``article_page`` figure that stacks all four at their real size inside the outline of
a journal page, to check they fit the space they are meant for.

**The lung slide is drawn a quarter turn clockwise.** It is a 1:3.9 strip, and a row of eight
upright panels of it would be 87 mm tall on its own; turned on its side the same row is
6 mm tall. The rotation is a pure quarter turn, never a transpose, so the tissue is not
mirrored. Breast is 1.33:1 and is drawn upright. Both keep the same layout, so the two rows
differ in panel height only.

Everything is drawn from the seed-averaged **raw** predictions of the combination the
gridsearch elected; see :mod:`runs`.

Usage::

    python plots.py                 # every figure, PNG and SVG, into figures/
    python plots.py --sample lung   # one sample
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

import runs  # noqa: E402
from runs import SAMPLES  # noqa: E402
from runs import Sample  # noqa: E402

FIGURES = Path(__file__).resolve().parent / "figures"

#: Width of a full-width (two-column span) figure, in inches: 180 mm.
WIDTH_IN = 180.0 / 25.4

#: Height of the text block of a journal page, in inches: 240 mm.
PAGE_HEIGHT_IN = 240.0 / 25.4

#: Colour of the cells that are not of the cell type a panel is about. Pale on purpose: some
#: cell types are themselves light, and a darker grey would swallow them.
OTHER_GREY = "#ebebeb"

FORMATS = ("png", "svg")


# --------------------------------------------------------------------------- geometry


def _display_coordinates(x: np.ndarray, y: np.ndarray, rotate: bool, y_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maps slide coordinates to the coordinates a panel is drawn in.

    The panels keep the image convention, y downwards, so an unrotated slide is drawn as it
    is. The rotation is the quarter turn clockwise ``(x, y) -> (y_max - y, x)``, which is a
    rotation and not a reflection: the slide is turned, never mirrored.

    Args:
        x: Horizontal slide coordinates.
        y: Vertical slide coordinates.
        rotate: Whether to turn the slide a quarter turn clockwise.
        y_max: Largest vertical coordinate of the slide, the pivot of the rotation.

    Returns:
        The coordinates to draw at, as (horizontal, vertical).
    """

    if rotate:
        return y_max - y, x

    return x, y


def _panel_extent(analyzer, sample: Sample) -> Tuple[float, float, float, float]:
    """
    The bounding box every panel of a sample shares, in display coordinates.

    All the panels of a row show the same field of view, so that a cell sits at the same
    place in all of them and the panels can be compared by eye.

    Args:
        analyzer: The analyzer of the sample.
        sample: The sample.

    Returns:
        ``(left, right, bottom, top)``.
    """

    centroids = analyzer.centroids
    y_max = float(centroids["y"].max())
    dx, dy = _display_coordinates(centroids["x"].to_numpy(), centroids["y"].to_numpy(), sample.rotate, y_max)

    return float(dx.min()), float(dx.max()), float(dy.max()), float(dy.min())


def _frame(ax: plt.Axes, extent: Tuple[float, float, float, float]) -> None:
    """
    Puts a panel in the shared field of view and strips it of everything else.

    Args:
        ax: The axis.
        extent: ``(left, right, bottom, top)``.
    """

    left, right, bottom, top = extent
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.axis("off")


# --------------------------------------------------------------------------- panels


def _draw_spot_pies(
    ax: plt.Axes,
    analyzer,
    sample: Sample,
    extent: Tuple[float, float, float, float],
    pie_scale: float = 1.0,
) -> int:
    """
    Draws the proportions of every spot as a pie at the spot's position.

    One wedge per cell type present, its angle the proportion of that cell type, starting at
    twelve o'clock and following the columns of the table, so a cell type always sits at the
    same place -- the convention :func:`hedest.spots.export_spot_pies` uses for QuPath. The
    radius is the real spot radius of the simulation, read from the AnnData.

    Args:
        ax: The axis.
        analyzer: The analyzer of the sample.
        sample: The sample.
        extent: The shared field of view.
        pie_scale: Factor on the spot radius, for when the true one is too small to read.

    Returns:
        The number of spots drawn.
    """

    from hedest.dataset_utils import pp_prop
    from hedest.spots import spot_geometry

    table = pp_prop(analyzer.proportions.copy())
    geometry = spot_geometry(analyzer.adata, sample.name)
    centres = dict(zip(map(str, geometry.ids), np.asarray(geometry.centers)[:, :2]))
    radius = 0.5 * geometry.diameter * pie_scale

    y_max = float(analyzer.centroids["y"].max())
    wedges: List[Wedge] = []
    colours: List[Tuple[float, float, float]] = []
    drawn = 0

    for spot, row in table.iterrows():
        if spot not in centres:
            continue
        x, y = centres[spot]
        dx, dy = _display_coordinates(np.array([x]), np.array([y]), sample.rotate, y_max)
        start = 0.0
        for name, value in row.items():
            if not value > 0:  # also skips NaN
                continue
            end = min(1.0, start + value)
            # The y axis points downwards, so a wedge that sweeps counter-clockwise in data
            # coordinates reads clockwise on the panel: -90 deg is twelve o'clock.
            wedges.append(Wedge((float(dx[0]), float(dy[0])), radius, -90.0 + 360.0 * start, -90.0 + 360.0 * end))
            colours.append(analyzer.palette.rgb(name))
            start = end
        drawn += start > 0

    if wedges:
        ax.add_collection(PatchCollection(wedges, facecolors=colours, edgecolors="none", linewidths=0.0))

    _frame(ax, extent)

    return drawn


def _draw_one_celltype(
    ax: plt.Axes,
    analyzer,
    sample: Sample,
    cell_type: str,
    extent: Tuple[float, float, float, float],
    point_size: float,
) -> int:
    """
    Draws every predicted cell, with only the cells of one cell type coloured.

    The other cells are kept, in grey, rather than dropped: they are what makes the panel a
    statement about *where* a cell type sits rather than about how many there are, and they
    outline the tissue so a sparse type is not read as an empty slide.

    Args:
        ax: The axis.
        analyzer: The analyzer of the sample.
        sample: The sample.
        cell_type: The cell type to colour.
        extent: The shared field of view.
        point_size: Marker size.

    Returns:
        The number of cells of that cell type.
    """

    centroids = analyzer.centroids
    labels = analyzer.labels.reindex(centroids.index)
    y_max = float(centroids["y"].max())
    dx, dy = _display_coordinates(centroids["x"].to_numpy(), centroids["y"].to_numpy(), sample.rotate, y_max)

    # rasterized: a panel holds 10^5 points, and leaving them as vector paths makes a 120 MB
    # SVG that no document can open. The points become one embedded image at the save dpi;
    # the titles, the legend and the pies stay vector.
    selection = (labels == cell_type).to_numpy()
    ax.scatter(
        dx[~selection],
        dy[~selection],
        s=point_size,
        color=OTHER_GREY,
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
    )
    # The coloured cells are drawn slightly larger than the grey ones: at this scale a point
    # is well under a pixel, so a pale cell type would otherwise be lost in the background.
    ax.scatter(
        dx[selection],
        dy[selection],
        s=point_size * 1.8,
        color=analyzer.palette.rgb(cell_type),
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
    )

    _frame(ax, extent)

    return int(selection.sum())


# --------------------------------------------------------------------------- figures


def proportions_vs_predictions(
    analyzer,
    sample: Sample,
    width_in: float = WIDTH_IN,
    gap_in: float = 0.045,
    legend_in: float = 0.26,
    point_size: float = 0.05,
    pie_scale: float = 1.0,
    fontsize: float = 5.8,
) -> Figure:
    """
    One row of ``1 + n`` panels: the spot proportions, then one panel per cell type.

    The panel width is set by the figure width divided by ``1 + n``, so the panel *height*
    follows from the shape of the slide and is not a free parameter. The axes are placed by
    hand, because the row has to keep a known height in millimetres.

    Args:
        analyzer: The analyzer of the sample.
        sample: The sample.
        width_in: Figure width, in inches.
        gap_in: Space between two panels, in inches.
        legend_in: Height of the legend strip, in inches. The title strip is not a parameter:
            it is solved for from the number of lines the longest title wraps onto.
        point_size: Marker size of one cell.
        pie_scale: Factor on the spot radius.
        fontsize: Size of the panel titles, in points.

    Returns:
        The figure.
    """

    cell_types = list(analyzer.ct_list)
    n_panels = 1 + len(cell_types)

    left, right, bottom, top = _panel_extent(analyzer, sample)
    aspect = (right - left) / (bottom - top)  # bottom > top: the y axis points downwards

    # The titles are built before the layout, because the height of the title strip depends
    # on how many lines the longest cell-type name wraps onto.
    counts = analyzer.labels.value_counts()
    n_cells = max(1, len(analyzer.labels))
    titles = [f"Spot proportions\n{len(analyzer.proportions):,} spots"]
    for cell_type in cell_types:
        total = int(counts.get(cell_type, 0))
        titles.append(f"{_short(cell_type)}\n{total:,} ({100.0 * total / n_cells:.0f}%)")

    panel_w = (width_in - (n_panels - 1) * gap_in) / n_panels
    panel_h = panel_w / aspect
    n_lines = max(title.count("\n") + 1 for title in titles)
    title_in = 0.06 + 1.3 * n_lines * fontsize / 72.0
    height_in = panel_h + title_in + legend_in

    logger.info(
        f"{sample.label}: {n_panels} panels of {panel_w * 25.4:.1f} x {panel_h * 25.4:.1f} mm "
        f"(slide aspect {aspect:.2f}{', rotated' if sample.rotate else ''}) "
        f"-> figure {width_in * 25.4:.0f} x {height_in * 25.4:.1f} mm"
    )

    fig = plt.figure(figsize=(width_in, height_in))

    for position, title in enumerate(titles):
        axis_left = position * (panel_w + gap_in)
        ax = fig.add_axes([axis_left / width_in, legend_in / height_in, panel_w / width_in, panel_h / height_in])

        if position == 0:
            _draw_spot_pies(ax, analyzer, sample, (left, right, bottom, top), pie_scale=pie_scale)
        else:
            _draw_one_celltype(ax, analyzer, sample, cell_types[position - 1], (left, right, bottom, top), point_size)

        # Anchored at the top of the figure and growing downwards, so a title that wraps onto
        # an extra line cannot run off the top edge.
        fig.text(
            (axis_left + 0.5 * panel_w) / width_in,
            1.0 - 0.3 * (title_in - 1.3 * n_lines * fontsize / 72.0) / height_in,
            title,
            ha="center",
            va="top",
            fontsize=fontsize,
            linespacing=1.3,
        )

    handles, labels = analyzer.palette.legend_handles(cell_types, markersize=4)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(cell_types),
        fontsize=fontsize,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        borderpad=0.0,
    )

    return fig


def celltype_gallery(
    analyzer,
    sample: Sample,
    width_in: float = WIDTH_IN,
    num_rows: int = 4,
    num_cols: int = 6,
    gap_in: float = 0.08,
    fontsize: float = 6.0,
) -> Figure:
    """
    The nucleus crops of the cells predicted for each cell type, one block per type.

    The blocks hold the cells the model is most confident about, which is the selection
    :meth:`PredAnalyzer.plot_celltype_grid` makes. That method is not called directly because
    it forwards neither the gap between blocks nor the label size, and both have to be set
    here: at its defaults a row of seven blocks is 214 mm wide, past the width of a page.
    :func:`hedest.analysis.plots.plot_gallery` takes them, so the crop size can be solved for
    and the row spans exactly ``width_in``.

    Args:
        analyzer: The analyzer of the sample.
        sample: The sample.
        width_in: Figure width, in inches.
        num_rows: Rows of crops inside one block.
        num_cols: Columns of crops inside one block.
        gap_in: Space between two blocks, in inches.
        fontsize: Size of the block labels, in points.

    Returns:
        The figure.
    """

    from hedest.analysis.plots import plot_gallery

    cell_types = list(analyzer.ct_list)
    n_blocks = len(cell_types)
    capacity = max(1, num_rows) * max(1, num_cols)

    blocks = {}
    for cell_type in cell_types:
        cells = analyzer.labels.index[analyzer.labels == cell_type]
        cells = [cell for cell in cells if cell in analyzer.image_dict]
        if cells:
            blocks[cell_type] = list(analyzer.confidence.loc[cells].nlargest(capacity).index)

    # plot_gallery sizes itself: width = n * num_cols * cell_size + (n - 1) * gap.
    cell_size = (width_in - (n_blocks - 1) * gap_in) / (n_blocks * num_cols)
    counts = analyzer.labels.value_counts()

    logger.info(
        f"{sample.label}: gallery of {len(blocks)} blocks of {num_rows}x{num_cols} crops of "
        f"{cell_size * 25.4:.1f} mm -> blocks {num_cols * cell_size * 25.4:.1f} mm wide"
    )

    return plot_gallery(
        analyzer.image_dict,
        blocks,
        num_rows=num_rows,
        num_cols=num_cols,
        cell_size=cell_size,
        mosaic=(1, n_blocks),
        palette=analyzer.palette,
        counts={name: int(counts.get(name, 0)) for name in blocks},
        gap=gap_in,
        fontsize=fontsize,
    )


def _short(name: str) -> str:
    """
    Shortens a cell-type name for a panel title two centimetres wide.

    Args:
        name: The cell type.

    Returns:
        The name, with its separators turned into line breaks.
    """

    from hedest.analysis.plots import wrap_name

    return wrap_name(name.replace("/", "/_"), 14).replace("/_", "/")


def article_page(panels: Sequence[Tuple[str, Path]], budget_mm: float = 116.0) -> Figure:
    """
    Stacks the figures at their real size inside the outline of a journal page.

    This is a check, not a result: the figures are placed at the width and height they were
    saved at, so the page shows how much of it they actually take, against the line the
    budget allows.

    Args:
        panels: ``(label, png path)`` in the order they should appear.
        budget_mm: Height the four figures together are meant to fit inside, in millimetres.

    Returns:
        The figure.
    """

    import matplotlib.image as mpimg

    images = [(label, mpimg.imread(str(path))) for label, path in panels]
    fig = plt.figure(figsize=(WIDTH_IN, PAGE_HEIGHT_IN))
    fig.patch.set_facecolor("white")

    # Kept tight on purpose: the four figures are already 102 mm, so a generous letter strip
    # and generous gaps would on their own push the block past the budget line below.
    label_in = 0.085
    gap_in = 0.04
    cursor = PAGE_HEIGHT_IN

    for letter, image in images:
        rows, columns = image.shape[0], image.shape[1]
        panel_h = WIDTH_IN * rows / columns
        cursor -= label_in + panel_h
        ax = fig.add_axes([0.0, cursor / PAGE_HEIGHT_IN, 1.0, panel_h / PAGE_HEIGHT_IN])
        ax.imshow(image)
        ax.axis("off")
        fig.text(
            0.0,
            (cursor + panel_h + 0.12 * label_in) / PAGE_HEIGHT_IN,
            letter,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        cursor -= gap_in

    used_in = PAGE_HEIGHT_IN - cursor - gap_in
    budget_in = budget_mm / 25.4
    line = 1.0 - budget_in / PAGE_HEIGHT_IN
    fig.add_artist(
        plt.Line2D([0.0, 1.0], [line, line], color="#c0392b", linestyle=(0, (4, 3)), linewidth=0.8, zorder=5)
    )
    fig.text(
        1.0,
        line - 0.006,
        f"budget {budget_mm:.0f} mm ({budget_mm / (PAGE_HEIGHT_IN * 25.4):.0%} of the page) — "
        f"the four figures use {used_in * 25.4:.0f} mm",
        ha="right",
        va="top",
        fontsize=7,
        color="#c0392b",
    )

    logger.info(f"article page: the four figures use {used_in * 25.4:.1f} mm of a {PAGE_HEIGHT_IN * 25.4:.0f} mm page.")

    return fig


# --------------------------------------------------------------------------- driver


def _save(fig: Figure, name: str, tight: bool = False) -> Path:
    """
    Writes a figure in every format, into ``figures/``.

    Args:
        fig: The figure.
        name: File name without an extension.
        tight: Whether to let matplotlib shrink the margins. False everywhere the axes were
            placed by hand, which is every figure here: a tight bounding box would undo the
            millimetre sizes the layout was solved for.

    Returns:
        The path of the PNG.
    """

    FIGURES.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"} if tight else {}

    # 600 dpi for the SVG too: it is what the rasterized point layers are resampled at, so
    # dropping it there would leave a vector file with blurred cells in it.
    for extension in FORMATS:
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=600, **kwargs)
    plt.close(fig)

    logger.info(f"-> {name}.{'/'.join(FORMATS)}")

    return FIGURES / f"{name}.png"


def main(keys: Optional[Sequence[str]] = None) -> None:
    """
    Draws every figure of the requested samples, and the article page when both are drawn.

    Args:
        keys: The samples to draw. All of them when None.
    """

    keys = list(keys) if keys else list(SAMPLES)
    panels: List[Tuple[str, Path]] = []

    for index, key in enumerate(keys):
        sample = SAMPLES[key]
        logger.info(f"=== {sample.label} ===")
        analyzer = runs.load_analyzer(sample)

        figure = proportions_vs_predictions(analyzer, sample)
        panels.append((f"{chr(ord('a') + 2 * index)}", _save(figure, f"{key}_proportions_vs_predictions")))

        figure = celltype_gallery(analyzer, sample)
        panels.append((f"{chr(ord('a') + 2 * index + 1)}", _save(figure, f"{key}_celltype_gallery")))

        del analyzer

    if len(keys) == len(SAMPLES):
        _save(article_page(panels), "article_page")

    logger.info(f"Every figure is in {FIGURES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", action="append", choices=sorted(SAMPLES), help="Only this sample; repeatable.")
    arguments = parser.parse_args()
    main(arguments.sample)
