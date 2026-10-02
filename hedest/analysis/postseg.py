"""
Looking at a slide: the tissue, the segmentation, the spots and the predictions.

:class:`SlideVisualizer` is usable at any point of the workflow. Right after segmentation it
shows the nuclei; with an AnnData object it adds the spots; with cell labels, from HEDeST or
from anywhere else, it colours the nuclei by cell type.

It reads through OpenSlide directly, not through the vendored HoVer-Net handler, so the
analysis package does not depend on ``external/``. Two consequences worth knowing:

- the window is always given in **full-resolution (level 0) coordinates**, whatever
  resolution the region is actually read at;
- a large window is read from the pyramid level that is just fine enough for the requested
  output size, and single-level slides are read in tiles and downsampled, so asking for a
  whole-slide view of a 26 000 x 26 000 slide costs a few hundred megabytes, not a few
  gigabytes.
"""
from __future__ import annotations

import base64
import io
import math
from collections.abc import Sequence
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from anndata import AnnData
from loguru import logger
from matplotlib.figure import Figure
from PIL import Image
from scipy.spatial import Delaunay

from hedest.analysis.palette import OTHER_LABEL
from hedest.analysis.palette import Palette
from hedest.analysis.plots import close
from hedest.analysis.plots import plot_celltype_points
from hedest.slide import read_mpp
from hedest.spots import load_seg_dict
from hedest.spots import spot_geometry

# A full-slide read is capped at this many pixels on the longest side.
DEFAULT_MAX_PIXELS = 4000
# Size of the tiles used when a slide has no suitable pyramid level.
READ_TILE = 2048
# Below this many pixels per nucleus, contours are not worth drawing.
MIN_PIXELS_PER_NUCLEUS = 6.0

Window = Union[str, Tuple[Tuple[float, float], Tuple[float, float]]]


class Region:
    """
    A region of a slide, read at some downsample.

    Attributes:
        image: The pixels, as an ``(h, w, 3)`` uint8 array.
        origin: Top-left corner of the region, in level-0 coordinates.
        size: Width and height of the region, in level-0 pixels.
        downsample: How much the image is downsampled relative to level 0.
    """

    def __init__(self, image: np.ndarray, origin: Tuple[float, float], size: Tuple[float, float], downsample: float):
        self.image = image
        self.origin = (float(origin[0]), float(origin[1]))
        self.size = (float(size[0]), float(size[1]))
        self.downsample = float(downsample)

    def __repr__(self) -> str:
        return (
            f"Region(origin={self.origin}, size={self.size}, "
            f"downsample={self.downsample:.2f}, shape={self.image.shape})"
        )

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """``(left, right, bottom, top)`` in level-0 coordinates, for ``imshow``."""

        return (
            self.origin[0],
            self.origin[0] + self.size[0],
            self.origin[1] + self.size[1],
            self.origin[1],
        )

    def to_local(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Converts level-0 coordinates into pixel coordinates inside ``image``.

        Args:
            coordinates: An ``(n, 2)`` array of (x, y) positions.

        Returns:
            The positions in the region's own pixel grid.
        """

        return (np.asarray(coordinates, dtype="float64") - np.asarray(self.origin)) / self.downsample

    def contains(self, coordinates: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """
        Tells which points fall inside the region.

        Args:
            coordinates: An ``(n, 2)`` array of (x, y) level-0 positions.
            margin: Extra tolerance, in level-0 pixels.

        Returns:
            A boolean mask.
        """

        points = np.asarray(coordinates, dtype="float64")
        x0, y0 = self.origin
        x1, y1 = x0 + self.size[0], y0 + self.size[1]

        return (
            (points[:, 0] >= x0 - margin)
            & (points[:, 0] <= x1 + margin)
            & (points[:, 1] >= y0 - margin)
            & (points[:, 1] <= y1 + margin)
        )


class SlideVisualizer:
    """
    Draws a slide with any combination of segmentation, spots and cell labels on top.

    Attributes:
        dimensions: Full-resolution size of the slide, as (width, height).
        mpp: Microns per pixel, either the one given or the one read from the slide.
        cell_ids: Ids of the segmented nuclei, in segmentation-file order.
        centroids: ``(n_cells, 2)`` array of nucleus centroids.
        palette: The colours in use.
    """

    def __init__(
        self,
        slide_path: str,
        adata: Optional[AnnData] = None,
        adata_name: Optional[str] = None,
        seg: Optional[Union[str, Dict[str, Any]]] = None,
        labels: Optional[Union[Dict[str, str], pd.Series]] = None,
        palette: Optional[Palette] = None,
        mpp: Optional[float] = None,
    ) -> None:
        """
        Opens a slide and, when given, loads the segmentation and the spots.

        Args:
            slide_path: Path to the slide. Anything OpenSlide can read; if it cannot, run
                ``python hedest/pipeline.py check <slide> --convert`` first.
            adata: AnnData object holding the spatial transcriptomics data, for the spots.
            adata_name: Key under ``adata.uns['spatial']``. Inferred when there is only one.
            seg: Segmentation dictionary or path to a HoVer-Net style JSON file.
            labels: Cell type per cell id, for instance
                ``PredAnalyzer.labels`` or the HoVer-Net types. When given, nuclei are
                coloured by cell type instead of a single colour.
            palette: Colours to use. Built from the labels when not given.
            mpp: Microns per pixel. Read from the slide when not given, and used for the
                spot diameter and for the scale bars.
        """

        self.slide_path = slide_path
        try:
            self.slide = openslide.OpenSlide(slide_path)
        except openslide.OpenSlideError as err:
            raise ValueError(
                f"OpenSlide cannot read {slide_path}. Convert it first with "
                f"'python hedest/pipeline.py check {slide_path} --convert'."
            ) from err

        self.dimensions: Tuple[int, int] = self.slide.dimensions
        self.mpp = mpp if mpp is not None else read_mpp(self.slide)
        self._thumbnails: Dict[int, Region] = {}

        self.seg_dict: Optional[Dict[str, Any]] = None
        self.cell_ids: List[str] = []
        self.centroids = np.empty((0, 2), dtype="float64")
        self._contours: List[np.ndarray] = []
        self._seg_types: List[str] = []

        if seg is not None:
            self._load_segmentation(seg)

        self.adata = adata
        self.adata_name = adata_name
        self.spots = None
        if adata is not None:
            if adata_name is None:
                spatial = adata.uns.get("spatial", {})
                if len(spatial) != 1:
                    raise ValueError(
                        "adata_name is required when adata.uns['spatial'] does not hold exactly one sample "
                        f"(found {list(spatial)})."
                    )
                self.adata_name = next(iter(spatial))
            self.spots = spot_geometry(adata, self.adata_name, mpp=self.mpp)

        self.labels: Optional[Dict[str, str]] = None
        self.palette = palette
        if labels is not None:
            self.set_labels(labels, palette=palette)
        elif palette is None and self._seg_types:
            self.palette = Palette(sorted(set(self._seg_types)))

    def __repr__(self) -> str:
        parts = [f"slide={self.slide_path}", f"dimensions={self.dimensions}"]
        if self.mpp is not None:
            parts.append(f"mpp={self.mpp:.4f}")
        parts.append(f"cells={len(self.cell_ids)}")
        parts.append(f"spots={0 if self.spots is None else len(self.spots.ids)}")
        parts.append(f"labelled={'yes' if self.labels else 'no'}")

        return f"SlideVisualizer({', '.join(parts)})"

    def close_slide(self) -> None:
        """Releases the OpenSlide handle."""

        self.slide.close()

    # ------------------------------------------------------------------ loading

    def _load_segmentation(self, seg: Union[str, Dict[str, Any]]) -> None:
        """
        Loads the nuclei into flat arrays.

        Args:
            seg: Segmentation dictionary or path to a HoVer-Net style JSON file.
        """

        self.seg_dict = load_seg_dict(seg)
        nuc = self.seg_dict["nuc"]

        self.cell_ids = list(nuc.keys())
        self.centroids = np.asarray([nuc[cell_id]["centroid"] for cell_id in self.cell_ids], dtype="float64")
        self._contours = [np.asarray(nuc[cell_id]["contour"], dtype="float64") for cell_id in self.cell_ids]
        self._seg_types = [str(nuc[cell_id].get("type")) for cell_id in self.cell_ids]
        logger.info(f"Loaded {len(self.cell_ids)} nuclei from the segmentation.")

    def set_labels(
        self,
        labels: Union[Dict[str, str], pd.Series],
        palette: Optional[Palette] = None,
    ) -> "SlideVisualizer":
        """
        Attaches a cell type to each nucleus, so the overlays are coloured by cell type.

        Args:
            labels: Cell type per cell id.
            palette: Colours to use. Built from the cell types when not given.

        Returns:
            The visualizer, so calls can be chained.
        """

        self.labels = dict(labels.items()) if isinstance(labels, pd.Series) else dict(labels)

        if palette is not None:
            self.palette = palette
        elif self.palette is None:
            self.palette = Palette(sorted(set(self.labels.values())))

        if self.cell_ids:
            missing = len(self.cell_ids) - sum(cell_id in self.labels for cell_id in self.cell_ids)
            if missing:
                logger.info(f"{missing}/{len(self.cell_ids)} nuclei have no label, drawn as '{OTHER_LABEL}'.")

        return self

    def cell_types(self) -> List[str]:
        """
        Returns the cell type of every nucleus, in segmentation order.

        Falls back to the ``type`` field of the segmentation when no labels are attached.
        """

        if self.labels is not None:
            return [self.labels.get(cell_id, OTHER_LABEL) for cell_id in self.cell_ids]

        return list(self._seg_types)

    # ------------------------------------------------------------------ reading

    def _resolve_window(self, window: Window) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Turns a window argument into an origin and a size, clipped to the slide.

        Args:
            window: ``"full"`` for the whole slide, or ``((x, y), (w, h))`` in level-0
                coordinates.

        Returns:
            The origin and the size.
        """

        if isinstance(window, str):
            if window != "full":
                raise ValueError(f"The only accepted window keyword is 'full', got '{window}'.")
            return (0.0, 0.0), (float(self.dimensions[0]), float(self.dimensions[1]))

        (x, y), (w, h) = window
        x, y = max(0.0, float(x)), max(0.0, float(y))
        w = min(float(w), self.dimensions[0] - x)
        h = min(float(h), self.dimensions[1] - y)
        if w <= 0 or h <= 0:
            raise ValueError(f"The window {window} does not intersect the slide {self.dimensions}.")

        return (x, y), (w, h)

    def read(self, window: Window = "full", max_pixels: int = DEFAULT_MAX_PIXELS) -> Region:
        """
        Reads a region of the slide, downsampling it if it would otherwise be too large.

        Args:
            window: ``"full"`` or ``((x, y), (w, h))`` in level-0 coordinates.
            max_pixels: Cap on the longest side of the returned image. The region is read
                from the best pyramid level, then resized if that is still not enough.

        Returns:
            The region, carrying the downsample that was applied.
        """

        origin, size = self._resolve_window(window)
        target_downsample = max(1.0, max(size) / float(max_pixels))

        if window == "full" and int(max_pixels) in self._thumbnails:
            return self._thumbnails[int(max_pixels)]

        level = self.slide.get_best_level_for_downsample(target_downsample)
        level_downsample = float(self.slide.level_downsamples[level])

        level_origin = (int(round(origin[0])), int(round(origin[1])))
        level_size = (
            max(1, int(round(size[0] / level_downsample))),
            max(1, int(round(size[1] / level_downsample))),
        )

        if level_size[0] * level_size[1] > 4 * max_pixels * max_pixels:
            # Even the coarsest level is too big to hold at once: read it in tiles.
            image = self._read_tiled(level, level_origin, level_size, max_pixels)
            downsample = max(size) / max(image.shape[1], image.shape[0])
        else:
            tile = self.slide.read_region(level_origin, level, level_size).convert("RGB")
            scale = max(1.0, max(level_size) / float(max_pixels))
            if scale > 1.0:
                tile = tile.resize(
                    (max(1, int(level_size[0] / scale)), max(1, int(level_size[1] / scale))), Image.BILINEAR
                )
            image = np.asarray(tile)
            downsample = size[0] / image.shape[1]

        region = Region(image, origin, size, downsample)
        if window == "full":
            self._thumbnails[int(max_pixels)] = region

        return region

    def _read_tiled(
        self,
        level: int,
        level_origin: Tuple[int, int],
        level_size: Tuple[int, int],
        max_pixels: int,
    ) -> np.ndarray:
        """
        Reads a region tile by tile and downsamples each tile, to bound memory use.

        Args:
            level: Pyramid level to read from.
            level_origin: Top-left corner, in level-0 coordinates.
            level_size: Size to read, in pixels of ``level``.
            max_pixels: Cap on the longest side of the result.

        Returns:
            The assembled, downsampled image.
        """

        level_downsample = float(self.slide.level_downsamples[level])
        scale = max(1.0, max(level_size) / float(max_pixels))
        out_size = (max(1, int(level_size[0] / scale)), max(1, int(level_size[1] / scale)))
        logger.info(
            f"Reading {level_size[0]}x{level_size[1]} px in tiles from level {level} "
            f"and downsampling to {out_size[0]}x{out_size[1]}."
        )

        canvas = Image.new("RGB", out_size)
        n_cols = math.ceil(level_size[0] / READ_TILE)
        n_rows = math.ceil(level_size[1] / READ_TILE)

        for row in range(n_rows):
            for col in range(n_cols):
                x = col * READ_TILE
                y = row * READ_TILE
                w = min(READ_TILE, level_size[0] - x)
                h = min(READ_TILE, level_size[1] - y)

                location = (
                    int(round(level_origin[0] + x * level_downsample)),
                    int(round(level_origin[1] + y * level_downsample)),
                )
                tile = self.slide.read_region(location, level, (w, h)).convert("RGB")

                out_x, out_y = int(x / scale), int(y / scale)
                out_w = max(1, min(int(round(w / scale)), out_size[0] - out_x))
                out_h = max(1, min(int(round(h / scale)), out_size[1] - out_y))
                canvas.paste(tile.resize((out_w, out_h), Image.BILINEAR), (out_x, out_y))

        return np.asarray(canvas)

    def thumbnail(self, max_pixels: int = DEFAULT_MAX_PIXELS) -> Region:
        """
        Returns a whole-slide overview, computed once and cached.

        Args:
            max_pixels: Cap on the longest side.

        Returns:
            The region covering the whole slide.
        """

        return self.read("full", max_pixels=max_pixels)

    # ------------------------------------------------------------------ overlays

    def _cells_in(self, region: Region) -> np.ndarray:
        """
        Indices of the nuclei whose centroid falls in a region, with a small margin so
        nuclei straddling the border are kept.

        Args:
            region: The region of interest.

        Returns:
            An array of indices into ``cell_ids``.
        """

        if not len(self.centroids):
            return np.empty(0, dtype=int)

        return np.flatnonzero(region.contains(self.centroids, margin=50.0))

    def _draw_contours(
        self,
        region: Region,
        indices: np.ndarray,
        types: Sequence[str],
        thickness: int = 2,
        filled: Optional[Sequence[str]] = None,
        draw_dot: bool = False,
        on_white: bool = False,
    ) -> np.ndarray:
        """
        Draws nucleus contours onto a copy of the region's pixels.

        Args:
            region: The region to draw on.
            indices: Indices of the nuclei to draw.
            types: Cell type of every nucleus, in segmentation order.
            thickness: Contour line thickness.
            filled: Cell types to fill rather than outline.
            draw_dot: Whether to mark centroids with a dot.
            on_white: Whether to draw on a white background instead of the tissue.

        Returns:
            The overlay image.
        """

        overlay = np.full_like(region.image, 255) if on_white else region.image.copy()
        palette = self.palette or Palette(sorted({str(t) for t in types}))
        filled_set = set(filled or ())

        for index in indices:
            contour = region.to_local(self._contours[index])
            contour = np.round(contour).astype(np.int32).reshape(-1, 1, 2)
            cell_type = types[index]
            color = tuple(int(c) for c in palette.rgb255(cell_type))

            if cell_type in filled_set:
                cv2.drawContours(overlay, [contour], -1, color, -1)
            else:
                cv2.drawContours(overlay, [contour], -1, color, thickness)

            if draw_dot:
                centroid = region.to_local(self.centroids[index : index + 1])[0]
                cv2.circle(overlay, (int(centroid[0]), int(centroid[1])), 3, color, -1)

        return overlay

    def _add_spots(
        self,
        ax: plt.Axes,
        region: Region,
        spot_prop_df: Optional[pd.DataFrame] = None,
        color: str = "black",
        linewidth: float = 1.5,
    ) -> None:
        """
        Draws the spots on an axis, as circles or as pie charts of their proportions.

        Args:
            ax: Axis to draw on, in region-local pixel coordinates.
            region: The region being shown.
            spot_prop_df: Proportions per spot. When given, each spot becomes a pie chart.
            color: Colour of the circles.
            linewidth: Width of the circles.
        """

        if self.spots is None:
            logger.warning("No AnnData object was given, so there are no spots to draw.")
            return

        radius = self.spots.diameter / 2.0
        inside = region.contains(self.spots.centers, margin=radius)
        centers_local = region.to_local(self.spots.centers)
        radius_local = radius / region.downsample

        # The pies are coloured by the cell types of spot_prop_df, which are not necessarily
        # the ones this visualizer draws nuclei with (they may be HoVer-Net class indices).
        pie_palette = self.palette
        if spot_prop_df is not None and (
            pie_palette is None or not all(column in pie_palette for column in spot_prop_df.columns)
        ):
            pie_palette = Palette(list(spot_prop_df.columns))

        for index in np.flatnonzero(inside):
            spot_id = self.spots.ids[index]
            x, y = centers_local[index]

            if spot_prop_df is not None and spot_id in spot_prop_df.index:
                proportions = spot_prop_df.loc[spot_id]
                proportions = proportions[proportions > 0]
                if proportions.empty:
                    continue
                # A nested axis, placed in figure coordinates, is the only way to put a
                # real pie chart at a data position.
                center_display = ax.transData.transform((x, y))
                edge_display = ax.transData.transform((x + radius_local, y))
                radius_display = abs(edge_display[0] - center_display[0])
                figure = ax.figure
                width = 2 * radius_display / figure.bbox.width
                height = 2 * radius_display / figure.bbox.height
                pie_ax = figure.add_axes(
                    [
                        (center_display[0] - radius_display) / figure.bbox.width,
                        (center_display[1] - radius_display) / figure.bbox.height,
                        width,
                        height,
                    ]
                )
                pie_ax.pie(proportions.values, colors=pie_palette.color_list(list(proportions.index)), startangle=90)
                pie_ax.set_aspect("equal")
                pie_ax.axis("off")
            else:
                ax.add_patch(plt.Circle((x, y), radius_local, color=color, fill=False, linewidth=linewidth))

    # ------------------------------------------------------------------ plotting

    def plot_slide(
        self,
        window: Window = "full",
        show_spots: bool = False,
        spot_prop_df: Optional[pd.DataFrame] = None,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (11.0, 11.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        Plots the tissue, optionally with the spots on top.

        Args:
            window: ``"full"`` or ``((x, y), (w, h))`` in level-0 coordinates.
            show_spots: Whether to draw the spots.
            spot_prop_df: Proportions per spot; each spot is then drawn as a pie chart.
            max_pixels: Cap on the longest side of the region that is read.
            title: Optional title.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        region = self.read(window, max_pixels=max_pixels)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(region.image)
        if show_spots or spot_prop_df is not None:
            self._add_spots(ax, region, spot_prop_df=spot_prop_df)
        ax.axis("off")
        ax.set_title(title if title is not None else self._default_title(region), fontsize=11)

        # The pies are inset axes, which tight_layout cannot place; it has nothing to do
        # here anyway, the figure being one image with its axis off.
        return close(fig, savefig, tight=spot_prop_df is None)

    def plot_seg(
        self,
        window: Window = "full",
        show_spots: bool = False,
        draw_dot: bool = False,
        thickness: int = 2,
        filled: bool = False,
        legend: bool = True,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (11.0, 11.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        Plots the tissue with the nucleus contours on top, coloured by cell type when
        labels are attached.

        For a window so large that a nucleus covers barely a pixel, contours cannot be
        seen; :meth:`plot_celltype_map` is the right call there, and a warning says so.

        Args:
            window: ``"full"`` or ``((x, y), (w, h))`` in level-0 coordinates.
            show_spots: Whether to draw the spots.
            draw_dot: Whether to mark centroids with a dot.
            thickness: Contour line thickness.
            filled: Whether to fill the nuclei instead of outlining them.
            legend: Whether to add a cell-type legend.
            max_pixels: Cap on the longest side of the region that is read.
            title: Optional title.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        if self.seg_dict is None:
            raise ValueError("No segmentation was given: pass seg=... to SlideVisualizer.")

        region = self.read(window, max_pixels=max_pixels)
        if region.downsample > 30.0 / MIN_PIXELS_PER_NUCLEUS:
            logger.warning(
                f"At a downsample of {region.downsample:.0f} a nucleus is about "
                f"{30.0 / region.downsample:.1f} px wide; plot_celltype_map() is easier to read."
            )

        indices = self._cells_in(region)
        types = self.cell_types()
        overlay = self._draw_contours(
            region,
            indices,
            types,
            thickness=thickness,
            filled=list(self.palette.names) if (filled and self.palette) else None,
            draw_dot=draw_dot,
        )

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(overlay)
        if show_spots:
            self._add_spots(ax, region)
        ax.axis("off")
        ax.set_title(
            title if title is not None else f"{self._default_title(region)} — {len(indices)} nuclei", fontsize=11
        )

        if legend and self.palette is not None:
            shown = sorted({types[i] for i in indices})
            handles, labels = self.palette.legend_handles([ct for ct in self.palette.names if ct in shown])
            if handles:
                ax.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    fontsize=9,
                    frameon=False,
                    handletextpad=0.6,
                    labelspacing=0.8,
                )

        return close(fig, savefig)

    def plot_seg_overlays(
        self,
        window: Window = "full",
        cell_types: Optional[Sequence[str]] = None,
        max_cols: int = 4,
        draw_dot: bool = False,
        thickness: int = 2,
        scale_cells: float = 1.0,
        on_white: bool = True,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        panel_size: Tuple[float, float] = (5.0, 5.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        One panel per cell type, each highlighting that type and greying out the rest.

        Args:
            window: ``"full"`` or ``((x, y), (w, h))`` in level-0 coordinates.
            cell_types: Cell types to show. Defaults to those present in the window.
            max_cols: Maximum number of columns.
            draw_dot: Whether to mark centroids with a dot.
            thickness: Contour line thickness.
            scale_cells: Scales the contours around their centroid, to make small nuclei
                visible in a large window.
            on_white: Whether to draw on white instead of the tissue.
            max_pixels: Cap on the longest side of the region that is read.
            panel_size: Size of one panel, in inches.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        if self.seg_dict is None:
            raise ValueError("No segmentation was given: pass seg=... to SlideVisualizer.")

        region = self.read(window, max_pixels=max_pixels)
        indices = self._cells_in(region)
        types = self.cell_types()
        present = sorted({types[i] for i in indices})
        shown = [ct for ct in (cell_types if cell_types is not None else present) if ct in present]

        if not shown:
            raise ValueError("None of the requested cell types is present in this window.")

        n_cols = min(len(shown), max_cols)
        n_rows = math.ceil(len(shown) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * panel_size[0], n_rows * panel_size[1]), squeeze=False
        )
        for ax in axes.flat:
            ax.axis("off")

        scaled_contours = None
        if scale_cells != 1.0:
            scaled_contours = self._contours
            self._contours = [
                (contour - self.centroids[i]) * scale_cells + self.centroids[i]
                for i, contour in enumerate(self._contours)
            ]

        try:
            for position, cell_type in enumerate(shown):
                ax = axes.flat[position]
                selection = np.array([i for i in indices if types[i] == cell_type], dtype=int)
                others = np.array([i for i in indices if types[i] != cell_type], dtype=int)

                overlay = np.full_like(region.image, 255) if on_white else region.image.copy()
                base = Region(overlay, region.origin, region.size, region.downsample)
                if len(others):
                    grey = [OTHER_LABEL] * len(types)
                    overlay = self._draw_contours(base, others, grey, thickness=1, on_white=False)
                    base = Region(overlay, region.origin, region.size, region.downsample)
                if len(selection):
                    overlay = self._draw_contours(
                        base, selection, types, thickness=thickness, filled=[cell_type], draw_dot=draw_dot
                    )

                ax.imshow(overlay)
                ax.set_title(f"{cell_type} ({len(selection)})", fontsize=13)
        finally:
            if scaled_contours is not None:
                self._contours = scaled_contours

        return close(fig, savefig)

    def plot_celltype_map(
        self,
        window: Window = "full",
        background: bool = True,
        point_size: Optional[float] = None,
        alpha: float = 0.9,
        cell_types: Optional[Sequence[str]] = None,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (12.0, 12.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        Every cell as a coloured dot at its position: the readable way to see cell types
        over a whole slide.

        Args:
            window: ``"full"`` or ``((x, y), (w, h))`` in level-0 coordinates.
            background: Whether to draw the tissue underneath.
            point_size: Marker size. Chosen from the cell density when not given.
            alpha: Marker transparency.
            cell_types: Restricts the map to these cell types.
            max_pixels: Cap on the longest side of the background image.
            title: Optional title.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        if not len(self.centroids):
            raise ValueError("No segmentation was given: pass seg=... to SlideVisualizer.")

        region = self.read(window, max_pixels=max_pixels)
        indices = self._cells_in(region)
        types = np.asarray(self.cell_types())

        if cell_types is not None:
            keep = np.isin(types[indices], list(cell_types))
            indices = indices[keep]

        if point_size is None:
            # Aim for markers that touch but do not merge: the area of the view divided by
            # the number of cells gives the pixel budget of one cell.
            per_cell = (region.image.shape[0] * region.image.shape[1]) / max(len(indices), 1)
            point_size = float(np.clip(per_cell / 12.0, 0.4, 24.0))

        palette = self.palette or Palette(sorted(set(types.tolist())))

        return plot_celltype_points(
            coordinates=region.to_local(self.centroids[indices]),
            labels=types[indices],
            palette=palette,
            background=region.image if background else None,
            extent=(0, region.image.shape[1], region.image.shape[0], 0) if background else None,
            point_size=point_size,
            alpha=alpha,
            title=title if title is not None else f"{self._default_title(region)} — {len(indices)} cells",
            figsize=figsize,
            savefig=savefig,
        )

    def plot_spot(
        self,
        spot_id: Optional[str] = None,
        margin: float = 40.0,
        draw_seg: Optional[bool] = None,
        spot_prop_df: Optional[pd.DataFrame] = None,
        title: Optional[str] = None,
        legend: bool = True,
        figsize: Tuple[float, float] = (7.0, 7.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        Zooms on one spot, with its outline and, when available, the segmentation.

        Args:
            spot_id: The spot to show. A random one is picked when None.
            margin: Extra pixels around the spot.
            draw_seg: Whether to draw the nuclei. Defaults to True when a segmentation is
                loaded.
            spot_prop_df: Proportions per spot; the spot is then drawn as a pie chart.
            title: Title of the panel. Pass an empty string for none.
            legend: Whether to add a cell-type legend.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        if self.spots is None:
            raise ValueError("No AnnData object was given: pass adata=... to SlideVisualizer.")

        if spot_id is None:
            spot_id = str(np.random.choice(self.spots.ids))
            logger.info(f"Randomly selected spot {spot_id}.")
        if spot_id not in self.spots.ids:
            raise ValueError(f"Spot {spot_id} is not in this AnnData object.")

        index = self.spots.ids.index(spot_id)
        center = self.spots.centers[index]
        side = self.spots.diameter + 2 * margin
        window = ((center[0] - side / 2, center[1] - side / 2), (side, side))

        if draw_seg is None:
            draw_seg = self.seg_dict is not None

        if title is None:
            title = f"spot {spot_id}"
        if draw_seg:
            return self.plot_seg(
                window,
                show_spots=spot_prop_df is None,
                max_pixels=int(side),
                title=title,
                legend=legend,
                figsize=figsize,
                savefig=savefig,
            )

        return self.plot_slide(
            window,
            show_spots=True,
            spot_prop_df=spot_prop_df,
            max_pixels=int(side),
            title=title,
            figsize=figsize,
            savefig=savefig,
        )

    def plot_delaunay_graph(
        self,
        window: Window = "full",
        max_distance: Optional[float] = None,
        linewidth: float = 0.4,
        color: str = "red",
        max_pixels: int = DEFAULT_MAX_PIXELS,
        figsize: Tuple[float, float] = (11.0, 11.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        Plots the Delaunay graph of the nuclei of a window, over the tissue.

        Args:
            window: ``"full"`` or ``((x, y), (w, h))`` in level-0 coordinates.
            max_distance: Drops edges longer than this, in level-0 pixels.
            linewidth: Width of the edges.
            color: Colour of the edges.
            max_pixels: Cap on the longest side of the region that is read.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        if not len(self.centroids):
            raise ValueError("No segmentation was given: pass seg=... to SlideVisualizer.")

        region = self.read(window, max_pixels=max_pixels)
        indices = self._cells_in(region)
        if len(indices) < 3:
            raise ValueError(f"Only {len(indices)} nuclei in this window: not enough for a triangulation.")

        coordinates = self.centroids[indices]
        local = region.to_local(coordinates)
        triangulation = Delaunay(coordinates)

        segments = set()
        for simplex in triangulation.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                segments.add((min(a, b), max(a, b)))

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(region.image)

        lines = []
        for a, b in segments:
            if max_distance is not None and np.linalg.norm(coordinates[a] - coordinates[b]) > max_distance:
                continue
            lines.append([local[a], local[b]])

        if lines:
            from matplotlib.collections import LineCollection

            ax.add_collection(LineCollection(lines, colors=color, linewidths=linewidth, alpha=0.6))

        ax.axis("off")
        ax.set_title(f"Delaunay graph — {len(indices)} nuclei, {len(lines)} edges", fontsize=11)

        return close(fig, savefig)

    def plot_interactive(
        self,
        window: Window = "full",
        draw_seg: Optional[bool] = None,
        show_spots: bool = False,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        height: int = 800,
    ) -> Any:
        """
        The same view as :meth:`plot_seg`, as a zoomable plotly figure with the cell type
        and cell id in the hover text.

        Args:
            window: ``"full"`` or ``((x, y), (w, h))`` in level-0 coordinates.
            draw_seg: Whether to draw the nuclei. Defaults to True when a segmentation is
                loaded.
            show_spots: Whether to draw the spots.
            max_pixels: Cap on the longest side of the region that is read.
            height: Height of the figure, in pixels.

        Returns:
            A ``plotly.graph_objects.Figure``.
        """

        import plotly.graph_objects as go

        region = self.read(window, max_pixels=max_pixels)
        if draw_seg is None:
            draw_seg = self.seg_dict is not None

        figure = go.Figure()
        figure.add_layout_image(
            dict(
                source=_to_data_uri(region.image),
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=region.image.shape[1],
                sizey=region.image.shape[0],
                sizing="stretch",
                layer="below",
            )
        )

        if draw_seg and len(self.centroids):
            indices = self._cells_in(region)
            types = np.asarray(self.cell_types())
            local = region.to_local(self.centroids[indices])
            palette = self.palette or Palette(sorted(set(types.tolist())))

            for cell_type in sorted(set(types[indices].tolist())):
                selection = types[indices] == cell_type
                red, green, blue = palette.rgb255(cell_type)
                figure.add_trace(
                    go.Scattergl(
                        x=local[selection, 0],
                        y=local[selection, 1],
                        mode="markers",
                        marker=dict(size=5, color=f"rgb({red},{green},{blue})"),
                        name=cell_type,
                        text=[f"cell {self.cell_ids[i]}" for i in indices[selection]],
                        hovertemplate="%{text}<br>" + cell_type + "<extra></extra>",
                    )
                )

        if show_spots and self.spots is not None:
            radius = self.spots.diameter / 2.0
            inside = region.contains(self.spots.centers, margin=radius)
            centers = region.to_local(self.spots.centers[inside])
            figure.add_trace(
                go.Scattergl(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode="markers",
                    marker=dict(
                        size=2 * radius / region.downsample,
                        color="rgba(0,0,0,0)",
                        line=dict(color="black", width=1),
                    ),
                    name="spots",
                    text=[self.spots.ids[i] for i in np.flatnonzero(inside)],
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        figure.update_xaxes(range=[0, region.image.shape[1]], visible=False, constrain="domain")
        figure.update_yaxes(
            range=[region.image.shape[0], 0], visible=False, scaleanchor="x", scaleratio=1, constrain="domain"
        )
        figure.update_layout(
            height=height, margin=dict(l=0, r=0, t=30, b=0), title=self._default_title(region), dragmode="pan"
        )

        return figure

    def _default_title(self, region: Region) -> str:
        """
        Builds a title saying what is being shown and at which resolution.

        Args:
            region: The region being shown.
        """

        name = self.adata_name or self.slide_path.rsplit("/", 1)[-1]
        if region.downsample > 1.2:
            return f"{name} — {int(region.size[0])}x{int(region.size[1])} px at 1/{region.downsample:.0f}"

        return f"{name} — {int(region.size[0])}x{int(region.size[1])} px"


def _to_data_uri(image: np.ndarray) -> str:
    """
    Encodes an image as a PNG data URI, which is how plotly embeds a background.

    Args:
        image: The image.

    Returns:
        The data URI.
    """

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")

    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


__all__ = ["Region", "SlideVisualizer"]
