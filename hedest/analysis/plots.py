"""
Stateless matplotlib helpers.

Nothing here reads a slide or knows about a run: every function takes plain arrays, frames
and dictionaries, so the helpers can be reused for figures that have nothing to do with
HEDeST. The higher-level compositions live in :mod:`hedest.analysis.pred_analyzer`.

Two conventions hold throughout the analysis package:

- a function that draws a whole figure **returns a closed figure** and never displays it,
  so a notebook shows it exactly once, as the value of the cell;
- a function that draws one panel takes ``ax`` and returns that ``ax``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from hedest.analysis.palette import Palette


def close(fig: Figure, savefig: Optional[str] = None, dpi: int = 200, tight: bool = True) -> Figure:
    """
    Saves a figure if asked, closes it and returns it.

    Closing matters: the inline notebook backend displays every figure still open at the
    end of a cell, so a figure that is both open and returned would appear twice.

    Args:
        fig: The figure to finalise.
        savefig: Path to write the figure to, if any.
        dpi: Resolution used when saving.
        tight: Whether to run ``tight_layout``. Figures whose axes are placed by hand, such
            as the galleries, pass False: tight_layout would reintroduce the gutters.

    Returns:
        The figure.
    """

    if tight:
        fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return fig


def as_image(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Turns a cell crop into something ``imshow`` accepts.

    Handles torch tensors and numpy arrays, channel-first and channel-last layouts, and
    both float and 8-bit ranges.

    Args:
        image: One cell image.

    Returns:
        An ``(h, w, 3)`` (or ``(h, w)``) array, float images clipped to [0, 1].
    """

    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[0] < array.shape[-1]:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    if np.issubdtype(array.dtype, np.floating):
        if array.max() > 1.0:
            array = array / 255.0
        array = np.clip(array, 0.0, 1.0)

    return array


def plot_pie_chart(
    ax: plt.Axes,
    data: pd.Series,
    palette: Palette,
    autopct: Optional[str] = "%1.0f%%",
    min_pct_label: float = 5.0,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Draws the composition of one spot, or of a whole slide, as a pie chart.

    Args:
        ax: Axis to draw on.
        data: Proportions indexed by cell type. They do not have to sum to one.
        palette: Colours to use.
        autopct: Percentage format, or None for no percentages.
        min_pct_label: Percentages below this value are left out, which keeps a pie with
            many small slices readable.
        title: Optional title.

    Returns:
        The axis.
    """

    data = data[data > 0]
    if data.empty:
        ax.text(0.5, 0.5, "no cells", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return ax

    def fmt(pct: float) -> str:
        return f"{pct:.0f}%" if pct >= min_pct_label else ""

    ax.pie(
        data.values,
        colors=palette.color_list(list(data.index)),
        autopct=None if autopct is None else fmt,
        startangle=90,
        textprops={"fontsize": 8},
    )
    ax.axis("equal")
    if title is not None:
        ax.set_title(title, fontsize=10)

    return ax


def plot_legend(
    palette: Palette,
    names: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    ncol: int = 1,
    fontsize: int = 12,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (3.0, 4.0),
    savefig: Optional[str] = None,
) -> Union[plt.Axes, Figure]:
    """
    Draws a standalone cell-type legend.

    Args:
        palette: Colours to use.
        names: Cell types to show. Defaults to the whole palette.
        ax: Axis to draw on. When None, a figure is created and returned instead.
        ncol: Number of legend columns.
        fontsize: Label font size.
        title: Optional legend title.
        figsize: Figure size, when this function creates the figure.
        savefig: Path to write the figure to, when this function creates it.

    Returns:
        The axis it drew on, or the figure it created.
    """

    handles, labels = palette.legend_handles(names)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.legend(handles, labels, loc="center", fontsize=fontsize, ncol=ncol, frameon=False, title=title)

    if not created:
        return ax

    return close(ax.figure, savefig)


def plot_history(
    history_train: Sequence[float],
    history_val: Sequence[float],
    figsize: Tuple[float, float] = (7.0, 4.0),
    savefig: Optional[str] = None,
) -> Figure:
    """
    Plots the training and validation loss, on one pair of axes so they can be compared.

    Args:
        history_train: Training loss per epoch.
        history_val: Validation loss per epoch.
        figsize: Figure size.
        savefig: Path to write the figure to.

    Returns:
        The figure.
    """

    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, len(history_train) + 1)
    ax.plot(epochs, history_train, color="#1f77b4", label="train")
    ax.plot(range(1, len(history_val) + 1), history_val, color="#d62728", label="validation")

    if len(history_val):
        best = int(np.argmin(history_val))
        ax.axvline(best + 1, color="grey", linestyle=":", linewidth=1)
        ax.plot([best + 1], [history_val[best]], "o", color="#d62728", markersize=5)
        ax.set_title(f"best epoch {best + 1} — validation loss {history_val[best]:.4f}", fontsize=10)

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)

    return close(fig, savefig)


def plot_cell(
    image: Union[torch.Tensor, np.ndarray],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    title_color: str = "black",
) -> plt.Axes:
    """
    Draws one cell crop.

    Args:
        image: The crop.
        ax: Axis to draw on. A new one is created if None.
        title: Optional title.
        title_color: Colour of the title, used to flag mispredictions in red.

    Returns:
        The axis.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(1.5, 1.5))

    ax.imshow(as_image(image))
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=8, color=title_color)

    return ax


def plot_cell_mosaic(
    image_dict: Dict[str, Any],
    cell_ids: Sequence[str],
    labels: Optional[Dict[str, str]] = None,
    true_labels: Optional[Dict[str, str]] = None,
    probs: Optional[Dict[str, float]] = None,
    num_cols: int = 8,
    cell_size: float = 1.6,
    suptitle: Optional[str] = None,
    savefig: Optional[str] = None,
) -> Figure:
    """
    Draws a grid of cell crops, optionally titled with their predicted cell type.

    Args:
        image_dict: Cell crops, keyed by cell id. Cells missing from it are skipped.
        cell_ids: The cells to draw, in order.
        labels: Predicted cell type per cell id.
        true_labels: Ground-truth cell type per cell id. When given, a title that
            disagrees with the prediction is shown in red as ``predicted (true)``.
        probs: Probability to show under the label, per cell id.
        num_cols: Number of columns.
        cell_size: Size of one crop, in inches.
        suptitle: Optional figure title.
        savefig: Path to write the figure to.

    Returns:
        The figure. Empty (with a message) if none of the cells has a crop.
    """

    present = [cell_id for cell_id in cell_ids if cell_id in image_dict]

    if not present:
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.text(0.5, 0.5, "no cell crops available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return close(fig, savefig)

    num_cols = max(1, min(num_cols, len(present)))
    num_rows = math.ceil(len(present) / num_cols)
    extra = 0.45 if labels is not None or probs is not None else 0.05
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * cell_size, num_rows * (cell_size + extra)),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for position, cell_id in enumerate(present):
        ax = axes.flat[position]
        title, color = None, "black"

        if labels is not None:
            predicted = labels.get(cell_id, "?")
            title, color = predicted, "black"
            if true_labels is not None:
                truth = true_labels.get(cell_id)
                if truth is not None and truth != predicted:
                    title, color = f"{predicted}\n({truth})", "#d62728"
        if probs is not None and cell_id in probs:
            prob_text = f"{probs[cell_id]:.2f}"
            title = prob_text if title is None else f"{title}\n{prob_text}"

        plot_cell(image_dict[cell_id], ax=ax, title=title, title_color=color)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12)

    return close(fig, savefig)


def _fit_crop(image: Any, shape: Tuple[int, int]) -> np.ndarray:
    """
    Brings one crop to a common size and to 8-bit RGB, so crops can be pasted side by side.

    Args:
        image: One cell crop, in any of the layouts :func:`as_image` accepts.
        shape: The ``(height, width)`` every crop is brought to, by a centre crop then a
            centred white padding.

    Returns:
        An ``(height, width, 3)`` uint8 array.
    """

    array = as_image(image)
    if np.issubdtype(array.dtype, np.floating):
        array = (255 * array).round()
    array = array.astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    array = array[..., :3]

    height, width = shape
    array = array[
        max(0, (array.shape[0] - height) // 2) : max(0, (array.shape[0] - height) // 2) + height,
        max(0, (array.shape[1] - width) // 2) : max(0, (array.shape[1] - width) // 2) + width,
    ]

    fitted = np.full((height, width, 3), 255, dtype=np.uint8)
    top, left = (height - array.shape[0]) // 2, (width - array.shape[1]) // 2
    fitted[top : top + array.shape[0], left : left + array.shape[1]] = array

    return fitted


def _tile_crops(
    images: Sequence[Any],
    num_rows: int,
    num_cols: int,
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Pastes crops into a single ``num_rows x num_cols`` image, filled row by row.

    Composing the image is what makes a gallery truly flush: one axes per crop leaves a
    hairline seam wherever an axis boundary falls between two pixels of the figure.

    Args:
        images: The crops, at most ``num_rows * num_cols`` of them. Missing ones leave white.
        num_rows: Number of rows of crops.
        num_cols: Number of columns of crops.
        shape: The ``(height, width)`` of one crop.

    Returns:
        The composed image.
    """

    height, width = shape
    canvas = np.full((num_rows * height, num_cols * width, 3), 255, dtype=np.uint8)
    for index, image in enumerate(images[: num_rows * num_cols]):
        row, column = divmod(index, num_cols)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = _fit_crop(image, shape)

    return canvas


def _mosaic_shape(count: int, tile_aspect: float, target: float = 1.4) -> Tuple[int, int]:
    """
    Picks how many rows and columns of galleries make the most convenient mosaic.

    Every arrangement of ``count`` tiles is scored on how close it brings the whole figure
    to ``target`` times wider than tall, with a penalty for the slots an incomplete last
    row would leave empty, so a clean 3x4 wins over a 3x5 with three holes.

    Args:
        count: Number of galleries to place.
        tile_aspect: Width over height of one gallery; they all have the same size.
        target: Preferred width over height of the whole mosaic.

    Returns:
        ``(rows, columns)``.
    """

    best = (count, 1)
    best_score = math.inf
    for columns in range(1, max(count, 1) + 1):
        rows = math.ceil(count / columns)
        score = abs(math.log(columns * tile_aspect / (rows * target))) + 0.25 * (rows * columns - count)
        if score < best_score:
            best, best_score = (rows, columns), score

    return best


def wrap_name(name: str, width: int = 20) -> str:
    """
    Breaks a long cell-type name at its underscores, so a label stays narrow.

    Args:
        name: The cell type.
        width: Target width, in characters.

    Returns:
        The name with newlines inserted, the underscores kept.
    """

    lines: list = []
    current = ""
    for part in name.split("_"):
        candidate = part if not current else f"{current}_{part}"
        if current and len(candidate) > width:
            lines.append(f"{current}_")
            current = part
        else:
            current = candidate
    lines.append(current)

    return "\n".join(lines)


def plot_gallery(
    image_dict: Dict[str, Any],
    blocks: Mapping[str, Sequence[str]],
    num_rows: int = 8,
    num_cols: int = 8,
    cell_size: float = 0.32,
    mosaic: Optional[Tuple[int, int]] = None,
    palette: Optional[Palette] = None,
    counts: Optional[Mapping[str, int]] = None,
    gap: float = 0.3,
    fontsize: float = 9.0,
    savefig: Optional[str] = None,
) -> Figure:
    """
    Draws a mosaic of flush galleries, one gallery per group.

    A group gets a block of ``num_rows x num_cols`` crops that touch, with no gutter, so
    the morphologies can be compared at a glance; the blocks are separated by ``gap``,
    framed in the colour of their group and arranged on a grid whose shape is chosen to
    keep the figure close to landscape unless ``mosaic`` imposes one.

    The axes are placed by hand rather than through a gridspec, because the space a gridspec
    leaves between panels depends on the figure margins, so ``wspace=0`` alone does not
    guarantee that the crops touch.

    Args:
        image_dict: Cell crops, keyed by cell id. Cells missing from it are skipped.
        blocks: The groups to draw, in order, as ``{label: cell ids}``. A group with more
            than ``num_rows * num_cols`` cells is truncated, a shorter one leaves its last
            slots empty.
        num_rows: Number of rows of crops inside one gallery.
        num_cols: Number of columns of crops inside one gallery.
        cell_size: Side of one crop, in inches.
        mosaic: ``(rows, columns)`` of galleries. Chosen internally when None.
        palette: Used to frame each gallery in the colour of its group.
        counts: Total number of cells per group, shown under the label. Defaults to the
            number of crops drawn.
        gap: Space between two galleries, in inches.
        fontsize: Size of the gallery labels, in points.
        savefig: Path to write the figure to.

    Returns:
        The figure. Empty (with a message) if none of the groups has a crop.
    """

    num_rows, num_cols = max(1, num_rows), max(1, num_cols)
    capacity = num_rows * num_cols

    present = {
        label: [cell_id for cell_id in cell_ids if cell_id in image_dict][:capacity]
        for label, cell_ids in blocks.items()
    }
    present = {label: cell_ids for label, cell_ids in present.items() if cell_ids}

    if not present:
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.text(0.5, 0.5, "no cell crops available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return close(fig, savefig)

    # All the crops are brought to the size of the first one, which keeps the galleries
    # aligned even if a crop was clipped at the border of the slide.
    first = next(iter(present.values()))[0]
    shape = as_image(image_dict[first]).shape[:2]
    canvases = {
        label: _tile_crops([image_dict[cell_id] for cell_id in cell_ids], num_rows, num_cols, shape)
        for label, cell_ids in present.items()
    }

    # One gallery is num_cols crops wide, and as tall as its crops plus its label strip.
    tile_width = num_cols * cell_size
    block_height = num_rows * cell_size * shape[0] / shape[1]
    characters = max(8, int(72.0 * tile_width / (0.55 * fontsize)))
    titles = {}
    for label, cell_ids in present.items():
        total = len(cell_ids) if counts is None else int(counts.get(label, len(cell_ids)))
        titles[label] = f"{wrap_name(label, characters)}\n{total:,} cells"
    header = 0.10 + 1.35 * max(text.count("\n") + 1 for text in titles.values()) * fontsize / 72.0
    tile_height = header + block_height

    mosaic_rows, mosaic_cols = mosaic if mosaic is not None else _mosaic_shape(len(present), tile_width / tile_height)
    mosaic_rows, mosaic_cols = max(1, mosaic_rows), max(1, mosaic_cols)
    width = mosaic_cols * tile_width + (mosaic_cols - 1) * gap
    height = mosaic_rows * tile_height + (mosaic_rows - 1) * gap

    fig = plt.figure(figsize=(width, height))

    for position, (label, canvas) in enumerate(canvases.items()):
        tile_row, tile_column = divmod(position, mosaic_cols)
        left = tile_column * (tile_width + gap)
        # Tiles are filled from the top of the figure, matplotlib measuring from the bottom.
        bottom = height - (tile_row + 1) * tile_height - tile_row * gap

        ax = fig.add_axes([left / width, bottom / height, tile_width / width, block_height / height])
        ax.imshow(canvas, aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black" if palette is None else palette.rgb(label))
            spine.set_linewidth(1.8)

        fig.text(
            (left + 0.5 * tile_width) / width,
            (bottom + block_height + 0.5 * header) / height,
            titles[label],
            ha="center",
            va="center",
            linespacing=1.35,
            fontsize=fontsize,
            fontweight="bold",
        )

    return close(fig, savefig, tight=False)


def plot_probability_histograms(
    predictions: pd.DataFrame,
    bins: int = 60,
    ncols: int = 3,
    log: bool = True,
    truth: Optional[pd.Series] = None,
    palette: Optional[Palette] = None,
    panel_size: Tuple[float, float] = (4.0, 2.6),
    savefig: Optional[str] = None,
) -> Figure:
    """
    One histogram of predicted probabilities per cell type.

    Args:
        predictions: Cells x cell types probability table.
        bins: Number of bins.
        ncols: Number of columns of panels.
        log: Whether to use a log y axis, which is almost always needed since most cells
            have a near-zero probability for most cell types.
        truth: Optional ground-truth cell type per cell, which splits each histogram into
            the cells that are of that type and the cells that are not.
        palette: Colours to use.
        panel_size: Size of one panel, in inches.
        savefig: Path to write the figure to.

    Returns:
        The figure.
    """

    cell_types = list(predictions.columns)
    ncols = max(1, min(ncols, len(cell_types)))
    nrows = math.ceil(len(cell_types) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * panel_size[0], nrows * panel_size[1]), squeeze=False, sharex=True
    )

    for ax in axes.flat:
        ax.axis("off")

    edges = np.linspace(0.0, 1.0, bins + 1)
    for position, cell_type in enumerate(cell_types):
        ax = axes.flat[position]
        ax.axis("on")
        color = palette.rgb(cell_type) if palette is not None else "#4c72b0"
        values = predictions[cell_type].to_numpy()

        if truth is None:
            ax.hist(values, bins=edges, color=color)
        else:
            is_type = truth.reindex(predictions.index).eq(cell_type).to_numpy()
            ax.hist(values[is_type], bins=edges, color=color, histtype="step", lw=1.6, label=cell_type)
            ax.hist(values[~is_type], bins=edges, color="grey", histtype="step", lw=1.2, label="other")
            ax.legend(fontsize=7, frameon=False)

        if log:
            ax.set_yscale("log")
        ax.set_title(cell_type, fontsize=10)
        ax.set_xlim(0, 1)

    for ax in axes[-1]:
        ax.set_xlabel("predicted probability")

    return close(fig, savefig)


def plot_proportion_scatter(
    true_proportions: pd.DataFrame,
    predicted_proportions: pd.DataFrame,
    palette: Optional[Palette] = None,
    ncols: int = 4,
    point_size: float = 4.0,
    alpha: float = 0.25,
    panel_size: Tuple[float, float] = (2.8, 2.8),
    savefig: Optional[str] = None,
) -> Figure:
    """
    Predicted against deconvoluted proportion, one panel per cell type.

    This is the main sanity check of a run without ground truth: HEDeST is trained to
    reproduce the spot proportions, so the points should follow the diagonal.

    Args:
        true_proportions: Spots x cell types, the deconvolution output.
        predicted_proportions: Spots x cell types, the mean of the cell predictions per spot.
        palette: Colours to use.
        ncols: Number of columns of panels.
        point_size: Marker size.
        alpha: Marker transparency.
        panel_size: Size of one panel, in inches.
        savefig: Path to write the figure to.

    Returns:
        The figure. Each panel carries its Pearson r.
    """

    truth, prediction = true_proportions.align(predicted_proportions, join="inner", axis=0)
    truth, prediction = truth.align(prediction, join="inner", axis=1)
    cell_types = list(truth.columns)

    ncols = max(1, min(ncols, len(cell_types)))
    nrows = math.ceil(len(cell_types) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_size[0], nrows * panel_size[1]), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for position, cell_type in enumerate(cell_types):
        ax = axes.flat[position]
        ax.axis("on")
        x = truth[cell_type].to_numpy()
        y = prediction[cell_type].to_numpy()
        color = palette.rgb(cell_type) if palette is not None else "#4c72b0"

        ax.scatter(x, y, s=point_size, alpha=alpha, color=color, edgecolors="none")
        top = float(max(x.max(initial=0.0), y.max(initial=0.0), 1e-3))
        ax.plot([0, top], [0, top], color="grey", linewidth=0.8, linestyle="--")

        title = cell_type
        if x.std() > 0 and y.std() > 0:
            title = f"{cell_type}\nr = {np.corrcoef(x, y)[0, 1]:.2f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("deconvolution", fontsize=8)
    for row in axes:
        row[0].set_ylabel("HEDeST", fontsize=8)

    return close(fig, savefig)


def plot_abundance(
    counts: pd.Series,
    reference: Optional[pd.Series] = None,
    palette: Optional[Palette] = None,
    reference_label: str = "deconvolution",
    figsize: Tuple[float, float] = (7.0, 4.0),
    savefig: Optional[str] = None,
) -> Figure:
    """
    Slide composition as a bar chart, optionally against a reference composition.

    Args:
        counts: Fraction of cells per cell type, as predicted.
        reference: Optional reference fractions, typically the mean spot proportions, drawn
            as an outline behind the bars.
        palette: Colours to use.
        reference_label: Legend label for the reference.
        figsize: Figure size.
        savefig: Path to write the figure to.

    Returns:
        The figure.
    """

    order = list(counts.index)
    positions = np.arange(len(order))
    colors = palette.color_list(order) if palette is not None else None

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(positions, counts.reindex(order).to_numpy(), color=colors, label="HEDeST")

    if reference is not None:
        ax.bar(
            positions,
            reference.reindex(order).to_numpy(),
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            label=reference_label,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("fraction of cells")
    ax.legend(frameon=False, fontsize=9)

    return close(fig, savefig)


def plot_matrix(
    matrix: pd.DataFrame,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annot: bool = False,
    fmt: str = ".2f",
    cbar_label: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    mask_upper: bool = False,
    cell_size: float = 0.62,
    figsize: Optional[Tuple[float, float]] = None,
    annot_fontsize: Optional[float] = None,
    savefig: Optional[str] = None,
) -> Figure:
    """
    Draws a square matrix as a heatmap. Used for colocalization, neighbour distances and
    confusion matrices.

    The figure grows with the matrix and the annotations are sized from the width of a
    cell, so the numbers stay inside their box however many cell types there are. The title
    is left-aligned on the heatmap, so a long one runs over the empty margin rather than
    over the colour bar.

    Args:
        matrix: The matrix, with cell types as index and columns.
        cmap: Colormap.
        vmin: Lower end of the colour scale.
        vmax: Upper end of the colour scale.
        annot: Whether to write the values in the cells.
        fmt: Format of the annotations.
        cbar_label: Label of the colour bar.
        title: Optional title.
        xlabel: Label of the columns.
        ylabel: Label of the rows.
        mask_upper: Whether to hide the upper triangle, for symmetric matrices.
        cell_size: Side of one cell of the heatmap, in inches, used to size the figure.
        figsize: Explicit figure size, which overrides ``cell_size``.
        annot_fontsize: Explicit annotation font size. Computed from the cell width when
            None, which is what keeps the numbers inside the boxes.
        savefig: Path to write the figure to.

    Returns:
        The figure.
    """

    import seaborn as sns

    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1) if mask_upper else None
    n_rows, n_columns = matrix.shape

    if figsize is None:
        # The extra inches hold the tick labels, which are cell-type names, and the colour bar.
        label_room = 0.085 * max((len(str(name)) for name in matrix.index), default=10)
        figsize = (
            cell_size * n_columns + label_room + 2.2,
            cell_size * n_rows + label_room + 1.4,
        )

    if annot_fontsize is None:
        # A cell is 72 * cell_size points wide and a digit is about 0.55 point wide per
        # point of font size, so this is the largest size whose longest annotation still
        # fits in a box, with a margin.
        values = np.ravel(matrix.to_numpy(dtype="float64"))
        values = values[np.isfinite(values)]
        widest = max((len(format(value, fmt)) for value in values), default=4)
        annot_fontsize = float(np.clip(0.85 * 72.0 * cell_size / (0.55 * max(widest, 1)), 5.0, 10.0))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        square=True,
        linewidths=0.5,
        mask=mask,
        annot_kws={"fontsize": annot_fontsize},
        cbar_kws={"label": cbar_label, "shrink": 0.75, "pad": 0.02} if cbar_label else {"shrink": 0.75, "pad": 0.02},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel(xlabel if xlabel is not None else "", fontsize=10)
    ax.set_ylabel(ylabel if ylabel is not None else "", fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=12, loc="left", pad=10)

    return close(fig, savefig)


def plot_celltype_points(
    coordinates: np.ndarray,
    labels: Sequence[str],
    palette: Palette,
    background: Optional[np.ndarray] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    point_size: float = 1.0,
    alpha: float = 0.9,
    legend: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 11.0),
    savefig: Optional[str] = None,
) -> Figure:
    """
    Scatters cells at their position on the slide, coloured by cell type.

    This is the readable way to show a whole slide: at that scale a nucleus is a couple of
    pixels wide, so contours are invisible and points are not.

    Args:
        coordinates: ``(n_cells, 2)`` array of positions, as (x, y), in the coordinate
            system of ``extent``.
        labels: Cell type of each row of ``coordinates``.
        palette: Colours to use.
        background: Optional image to draw underneath, typically a slide thumbnail.
        extent: ``(left, right, bottom, top)`` for the background, in the same coordinates
            as ``coordinates``. Required when a background is given.
        point_size: Marker size.
        alpha: Marker transparency.
        legend: Whether to add a cell-type legend outside the axes.
        title: Optional title.
        figsize: Figure size.
        savefig: Path to write the figure to.

    Returns:
        The figure.
    """

    fig, ax = plt.subplots(figsize=figsize)

    if background is not None:
        if extent is None:
            raise ValueError("extent is required when a background image is given.")
        ax.imshow(background, extent=extent)
    else:
        ax.invert_yaxis()
        ax.set_aspect("equal")

    labels = np.asarray(labels)
    for cell_type in palette.names:
        selection = labels == cell_type
        if not selection.any():
            continue
        ax.scatter(
            coordinates[selection, 0],
            coordinates[selection, 1],
            s=point_size,
            color=palette.rgb(cell_type),
            alpha=alpha,
            edgecolors="none",
            label=cell_type,
        )

    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=12)
    if legend:
        handles, legend_labels = palette.legend_handles([ct for ct in palette.names if (labels == ct).any()])
        ax.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=9,
            frameon=False,
            markerscale=1.4,
            handletextpad=0.6,
            labelspacing=0.8,
        )

    return close(fig, savefig)


def plot_stacked_bars(
    frame: pd.DataFrame,
    palette: Palette,
    xlabel: str = "",
    ylabel: str = "fraction",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 4.5),
    savefig: Optional[str] = None,
) -> Figure:
    """
    Stacked composition bars, one bar per row of the frame.

    Args:
        frame: Rows are the groups, columns the cell types, values the fractions.
        palette: Colours to use.
        xlabel: Label of the x axis.
        ylabel: Label of the y axis.
        title: Optional title.
        figsize: Figure size.
        savefig: Path to write the figure to.

    Returns:
        The figure.
    """

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(frame))
    positions = np.arange(len(frame))

    for cell_type in frame.columns:
        values = frame[cell_type].to_numpy()
        ax.bar(positions, values, bottom=bottom, color=palette.rgb(cell_type), label=cell_type, width=0.8)
        bottom += values

    ax.set_xticks(positions)
    ax.set_xticklabels([str(index) for index in frame.index], rotation=40, ha="right", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, float(bottom.max()) if len(bottom) else 1.0)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)

    return close(fig, savefig)


def polygon_area(contour: Sequence[Sequence[float]]) -> float:
    """
    Area of a polygon, by the shoelace formula.

    Args:
        contour: The vertices, as (x, y) pairs.

    Returns:
        The area, in squared units of the input coordinates.
    """

    points = np.asarray(contour, dtype="float64")
    x, y = points[:, 0], points[:, 1]

    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_areas(contours: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    """
    Areas of many polygons.

    Args:
        contours: One contour per polygon.

    Returns:
        The areas, in the order of the input.
    """

    return np.array([polygon_area(contour) for contour in contours], dtype="float64")


__all__ = [
    "as_image",
    "close",
    "plot_abundance",
    "plot_cell",
    "plot_cell_mosaic",
    "plot_celltype_points",
    "plot_gallery",
    "plot_history",
    "plot_legend",
    "plot_matrix",
    "plot_pie_chart",
    "plot_probability_histograms",
    "plot_proportion_scatter",
    "plot_stacked_bars",
    "polygon_area",
    "polygon_areas",
    "wrap_name",
]
