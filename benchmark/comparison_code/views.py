"""Tissue views: the same cells typed by the ground truth and by each method, side by side.

Three kinds of figure, all five columns wide (ground truth, HistoCell, PanoSpace, HEDeST,
HEDeST + PPSA) and all coloured with the STHELAR palette
(``simulations/semi_simulations/STHELAR/palette.py``), so a colour means the same cell type
here, across levels, and in every other STHELAR figure:

    crops/featured           the windows of ``config.CROP_WINDOWS``, one row each
    crops/{sample}_{level}   the sample's windows at one level, one row each
    slides/{sample}_{level}  the whole slide, every cell a coloured dot

Only the cells every panel can type are drawn, which is the set the scores are computed on:
PanoSpace does not cover every nucleus, and drawing its gaps would read as a prediction
rather than an absence. The figure under each panel is its agreement with the ground truth
over the cells drawn.

The segmentation and the slide thumbnail are the expensive part, so a sample is opened once
and all its levels and both configurations are drawn from it.

    python benchmark/comparison_code/views.py                    # everything
    python benchmark/comparison_code/views.py --what featured
    python benchmark/comparison_code/views.py --sample skin_s4 --config gt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import collect as K  # noqa: E402
import config as C  # noqa: E402

sys.path.insert(0, C.REPO)  # the hedest package, for the slide viewer

from hedest.analysis import Palette  # noqa: E402
from hedest.analysis import SlideVisualizer  # noqa: E402

PANELS = ["Ground truth", "HistoCell", "PanoSpace", "HEDeST", "HEDeST + PPSA"]
Window = Tuple[Tuple[int, int], Tuple[int, int]]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save_fig(fig: plt.Figure, stem: str, dpi: int = 200) -> None:
    """
    Saves ``fig`` as ``stem.png`` and closes it.

    These figures are tissue images with one marker or filled contour per cell, so an ``svg``
    would carry the same pixels plus a vector path per nucleus: a whole-slide panel set runs
    to hundreds of megabytes. The accuracy figures of ``plots.py``, which are genuinely
    vector, keep their ``svg``.

    Args:
        fig: The figure.
        stem: Path without extension.
        dpi: Resolution.
    """

    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def label_sets(sample: str, level: str, config: str, seed: int) -> Dict[str, pd.Series]:
    """
    The cell type each panel draws.

    Args:
        sample: The sample name.
        level: The annotation level.
        config: ``"gt"`` or ``"deconv"``.
        seed: The seed whose predictions are drawn.

    Returns:
        ``{panel: labels}``, missing panels simply absent.
    """

    labels: Dict[str, pd.Series] = {"Ground truth": K.load_truth(sample, level)}

    for method in ("HistoCell", "PanoSpace"):
        table = K.load_competitor(method, sample, level, config, seed)
        if table is not None:
            labels[method] = K.argmax_labels(table)

    for method, table in K.load_hedest(sample, level, config, seed).items():
        labels[method] = K.argmax_labels(table)

    return labels


def dense_windows(centroids: np.ndarray, size: int, count: int) -> List[Window]:
    """
    The windows holding the most nuclei, kept apart from one another.

    A plain density rule: the slide is covered by a half-overlapping grid of ``size`` px
    windows, these are sorted by how many nuclei they hold, and windows are taken in that
    order while they do not touch one that is already taken.

    Args:
        centroids: ``(n, 2)`` cell centroids in slide coordinates.
        size: Side of a window, in pixels.
        count: How many to return.

    Returns:
        The windows, ``((x, y), (size, size))``.
    """

    step = size // 2
    x_max, y_max = centroids[:, 0].max(), centroids[:, 1].max()
    counts = []
    for x0 in range(0, int(x_max - size), step):
        inside_x = (centroids[:, 0] >= x0) & (centroids[:, 0] < x0 + size)
        if not inside_x.any():
            continue
        ys = centroids[inside_x, 1]
        for y0 in range(0, int(y_max - size), step):
            n = int(((ys >= y0) & (ys < y0 + size)).sum())
            if n:
                counts.append((n, x0, y0))

    chosen: List[Window] = []
    for _, x0, y0 in sorted(counts, reverse=True):
        if all(abs(x0 - x) >= size or abs(y0 - y) >= size for (x, y), _ in chosen):
            chosen.append(((x0, y0), (size, size)))
        if len(chosen) == count:
            break

    return chosen


class SampleViews:
    """
    One sample's slide and segmentation, held open to draw all its levels.

    Args:
        sample: The sample name.
        max_pixels: Longest side of the whole-slide background.
    """

    def __init__(self, sample: str, max_pixels: int = 1600) -> None:
        self.sample = sample
        self.viz = SlideVisualizer(
            slide_path=os.path.join(C.BENCH_ROOT, sample, "he.tiff"),
            seg=os.path.join(C.BENCH_ROOT, sample, "hovernet.json"),
        )
        self.max_pixels = max_pixels
        self.index = {cell: i for i, cell in enumerate(self.viz.cell_ids)}
        self._full = None

    def close(self) -> None:
        """Releases the slide handle."""

        self.viz.close_slide()

    @property
    def full_region(self):
        """The whole-slide background, read once."""

        if self._full is None:
            self._full = self.viz.read("full", max_pixels=self.max_pixels)

        return self._full

    def _prepare(self, level: str, config: str, seed: int):
        """Labels, palette and the cells every panel can type."""

        labels = label_sets(self.sample, level, config, seed)
        panels = [p for p in PANELS if p in labels]
        palette = Palette.from_colors(C.bench_palette(self.sample, level))

        shared = set(labels["Ground truth"].index)
        for series in labels.values():
            shared &= set(series.index)

        return labels, panels, palette, shared

    def _legend(self, fig: plt.Figure, palette: Palette, present: Sequence[str]) -> None:
        """One shared cell-type legend under the figure."""

        handles = [Patch(facecolor=palette.rgb(t), edgecolor="none", label=t) for t in present]
        fig.legend(
            handles=handles,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=min(len(present), 6),
            fontsize=9,
        )

    def crops(self, level: str, config: str, windows: Sequence[Window], stem: str, seed: int = 0) -> None:
        """
        One row per window, one column per panel, nuclei filled with their cell-type colour.

        Args:
            level: The annotation level.
            config: ``"gt"`` or ``"deconv"``.
            windows: The windows to draw.
            stem: Path without extension.
            seed: The seed whose predictions are drawn.
        """

        labels, panels, palette, shared = self._prepare(level, config, seed)
        truth = labels["Ground truth"]
        classes = list(palette.names)

        fig, axes = plt.subplots(
            len(windows), len(panels), figsize=(3.0 * len(panels), 3.15 * len(windows)), squeeze=False
        )

        drawn_types: set = set()
        for row, window in enumerate(windows):
            region = self.viz.read(window, max_pixels=900)
            indices = [i for i in self.viz._cells_in(region) if self.viz.cell_ids[i] in shared]
            cells = [self.viz.cell_ids[i] for i in indices]
            drawn_types |= set(truth.loc[cells]) if cells else set()

            for col, panel in enumerate(panels):
                ax = axes[row][col]
                self.viz.set_labels(labels[panel], palette=palette)
                ax.imshow(self.viz._draw_contours(region, indices, self.viz.cell_types(), filled=classes))
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_edgecolor("0.75")
                if row == 0:
                    ax.set_title(panel, fontsize=11)
                if panel == "Ground truth":
                    ax.set_xlabel(f"{len(cells)} nuclei", fontsize=9)
                elif cells:
                    agree = float((labels[panel].reindex(cells) == truth.loc[cells]).mean())
                    ax.set_xlabel(f"{agree:.0%} agreement", fontsize=9)

            (x, y), (w, h) = window
            axes[row][0].set_ylabel(f"x {x}, y {y}\n{w} x {h} px", fontsize=9)

        self._legend(fig, palette, [t for t in classes if t in drawn_types])
        fig.suptitle(f"{self.sample}, {level} - {C.CONFIGS[config]}, seed {seed}", fontsize=12)
        fig.subplots_adjust(wspace=0.04, hspace=0.14)
        save_fig(fig, stem)

    def slide(self, level: str, config: str, stem: str, seed: int = 0) -> None:
        """
        The whole slide, every cell a dot in its cell-type colour, one panel per method.

        Args:
            level: The annotation level.
            config: ``"gt"`` or ``"deconv"``.
            stem: Path without extension.
            seed: The seed whose predictions are drawn.
        """

        labels, panels, palette, shared = self._prepare(level, config, seed)
        truth = labels["Ground truth"]

        region = self.full_region
        rows = [self.index[c] for c in shared if c in self.index]
        cells = [self.viz.cell_ids[i] for i in rows]
        points = region.to_local(self.viz.centroids[rows])
        height, width = region.image.shape[:2]
        size = float(np.clip((height * width) / max(len(rows), 1) / 14.0, 0.25, 8.0))

        # The panels follow the slide's aspect ratio, so a tall section does not sit in a
        # wide box with white on both sides.
        panel_width = float(np.clip(5.0 * width / height, 1.8, 6.5))
        fig, axes = plt.subplots(1, len(panels), figsize=(panel_width * len(panels), 5.6), squeeze=False)
        for ax, panel in zip(axes[0], panels):
            ax.imshow(region.image)
            series = labels[panel].reindex(cells)
            colours = np.array([palette.rgb(t) for t in palette.names])
            code = pd.Categorical(series, categories=list(palette.names)).codes
            known = code >= 0
            ax.scatter(points[known, 0], points[known, 1], s=size, c=colours[code[known]], alpha=0.9, linewidths=0)
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(panel, fontsize=11)
            if panel == "Ground truth":
                ax.set_xlabel(f"{len(cells)} cells", fontsize=9)
            else:
                ax.set_xlabel(f"{float((series == truth.loc[cells]).mean()):.0%} agreement", fontsize=9)

        self._legend(fig, palette, [t for t in palette.names if (truth.loc[cells] == t).any()])
        fig.suptitle(f"{self.sample}, {level} - {C.CONFIGS[config]}, seed {seed}", fontsize=12)
        fig.subplots_adjust(wspace=0.03)
        save_fig(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=list(C.CONFIGS) + ["both"], default="both", help="Which configuration.")
    parser.add_argument("--sample", default=None, help="Only this sample.")
    parser.add_argument(
        "--what",
        choices=["all", "featured", "crops", "slides"],
        default="all",
        help="Which figures to draw.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed whose predictions are drawn.")
    parser.add_argument("--plot-dir", default=C.PLOT_DIR, help="Where the figures go.")
    args = parser.parse_args()

    configs = list(C.CONFIGS) if args.config == "both" else [args.config]
    samples = [args.sample] if args.sample else C.SAMPLES
    start = time.time()

    for sample in samples:
        work = {c: [lvl for s, lvl in C.units(c) if s == sample] for c in configs}
        if not any(work.values()):
            continue

        log(f"=== {sample}: opening slide and segmentation")
        views = SampleViews(sample)
        windows = dense_windows(views.viz.centroids, C.CROP_SIZE, C.CROPS_PER_SAMPLE)
        log(f"{sample}: {len(views.viz.cell_ids)} nuclei, windows {[w[0] for w in windows]}")

        try:
            for config in configs:
                out = os.path.join(args.plot_dir, config)
                for level in work[config]:
                    if args.what in ("all", "crops"):
                        views.crops(level, config, windows, os.path.join(out, "crops", f"{sample}_{level}"), args.seed)
                    if args.what in ("all", "slides"):
                        views.slide(level, config, os.path.join(out, "slides", f"{sample}_{level}"), args.seed)
                    if args.what in ("all", "featured"):
                        spec = C.CROP_WINDOWS.get(config)
                        if spec and spec["sample"] == sample and spec["level"] == level:
                            views.crops(
                                level, config, spec["windows"], os.path.join(out, "crops", "featured"), args.seed
                            )
                    log(f"{sample}/{level} ({config}) done")
        finally:
            views.close()

    log(f"Finished in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
