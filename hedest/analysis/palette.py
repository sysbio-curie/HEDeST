"""
One colour per cell type, shared by every plot, overlay and export.

``Palette`` replaces the two ad-hoc colour dictionary formats that used to travel around
the analysis code (``{name: rgba}`` for matplotlib and ``{"0": [name, [r, g, b, a]]}`` for
the HoVer-Net overlays and the QuPath GeoJSON). It holds one ordering of the cell types and
converts to whichever format a consumer needs, so a cell type keeps its colour across the
whole analysis.
"""
from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

# tab20 holds 20 distinct colours; the b/c variants extend it to 60 before anything repeats.
DEFAULT_CMAPS = ("tab20", "tab20b", "tab20c")
OTHER_COLOR = (0.63, 0.63, 0.63)
OTHER_LABEL = "Other"


class Palette:
    """
    A fixed mapping from cell type to colour.

    Attributes:
        names: The cell types, in the order they were given, which is also the order of the
            columns of the prediction tables and therefore the class indices of the model.
    """

    def __init__(self, names: Sequence[str], cmaps: Iterable[str] = DEFAULT_CMAPS) -> None:
        """
        Builds a palette for a list of cell types.

        Args:
            names: The cell types. The order matters: it defines the class indices.
            cmaps: Matplotlib colormap names to draw the colours from, in order.
        """

        self.names = list(names)
        self._index = {name: i for i, name in enumerate(self.names)}

        colors: List[Tuple[float, float, float]] = []
        for cmap_name in cmaps:
            cmap = plt.get_cmap(cmap_name)
            colors.extend(tuple(cmap(i)[:3]) for i in range(cmap.N))

        if len(self.names) > len(colors):
            # Falls back to a continuous colormap rather than repeating colours.
            cmap = plt.get_cmap("gist_ncar")
            colors = [tuple(cmap(i / max(len(self.names) - 1, 1))[:3]) for i in range(len(self.names))]

        self._colors = {name: colors[i] for i, name in enumerate(self.names)}

    @classmethod
    def from_colors(cls, colors: Mapping[str, Any], names: Optional[Sequence[str]] = None) -> "Palette":
        """
        Builds a palette from explicit colours, which is how a user imposes their own.

        Args:
            colors: ``{cell type: colour}``, where the colour is anything matplotlib
                accepts: ``"#e41a1c"``, ``"tab:blue"``, ``(0.9, 0.1, 0.1)``.
            names: The cell types, in the order that defines the class indices. Defaults to
                the order of ``colors``. A cell type listed here but absent from ``colors``
                keeps the colour the default palette would have given it.

        Returns:
            The palette.
        """

        order = list(names) if names is not None else list(colors)
        order += [name for name in colors if name not in order]

        palette = cls(order)
        palette._colors.update({name: to_rgb(value) for name, value in colors.items()})

        return palette

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __repr__(self) -> str:
        return f"Palette({len(self.names)} cell types: {', '.join(self.names)})"

    def rgb(self, name: str) -> Tuple[float, float, float]:
        """
        Returns the matplotlib colour of a cell type, as floats in [0, 1].

        Args:
            name: A cell type. Anything unknown gets the grey reserved for "other".
        """

        return self._colors.get(name, OTHER_COLOR)

    def rgb255(self, name: str) -> Tuple[int, int, int]:
        """
        Returns the colour of a cell type as 8-bit integers, for OpenCV and GeoJSON.

        Args:
            name: A cell type.
        """

        return tuple(int(round(255 * channel)) for channel in self.rgb(name))  # type: ignore[return-value]

    def index(self, name: str) -> int:
        """
        Returns the class index of a cell type.

        Args:
            name: A cell type.
        """

        return self._index[name]

    def name_of(self, index: int) -> str:
        """
        Returns the cell type of a class index.

        Args:
            index: A class index.
        """

        return self.names[index]

    @property
    def colors(self) -> Dict[str, Tuple[float, float, float]]:
        """The whole mapping, for the matplotlib and seaborn arguments that take a dict."""

        return dict(self._colors)

    def color_list(self, names: Optional[Sequence[str]] = None) -> List[Tuple[float, float, float]]:
        """
        Returns the colours of a series of cell types, in order.

        Args:
            names: The cell types to look up. Defaults to the whole palette.
        """

        return [self.rgb(name) for name in (self.names if names is None else names)]

    def as_hovernet(self, highlight: Optional[Sequence[str]] = None) -> Dict[str, Tuple[str, Tuple[int, int, int]]]:
        """
        Returns the ``{class index as str: (name, (r, g, b))}`` format used by the
        segmentation overlays and the QuPath GeoJSON export.

        Args:
            highlight: When given, only these cell types keep their colour; the others are
                greyed out and renamed, which is how the per-type overlay panels are drawn.
        """

        mapping = {}
        for i, name in enumerate(self.names):
            if highlight is not None and name not in highlight:
                mapping[str(i)] = (OTHER_LABEL, tuple(int(round(255 * c)) for c in OTHER_COLOR))
            else:
                mapping[str(i)] = (name, self.rgb255(name))

        return mapping  # type: ignore[return-value]

    def as_geojson_dict(self) -> Dict[str, list]:
        """
        Returns the YAML/JSON-serialisable colour dictionary written next to the GeoJSON
        exports, in the ``{"0": ["CellType", [r, g, b, a]]}`` shape QuPath tooling expects.
        """

        return {str(i): [name, list(self.rgb255(name)) + [255]] for i, name in enumerate(self.names)}

    def legend_handles(self, names: Optional[Sequence[str]] = None, markersize: int = 10) -> Tuple[list, List[str]]:
        """
        Builds matplotlib legend handles for the cell types.

        Args:
            names: The cell types to include. Defaults to the whole palette.
            markersize: Size of the legend markers.

        Returns:
            The handles and their labels, ready for ``ax.legend(*handles_and_labels)``.
        """

        labels = list(self.names if names is None else names)
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=self.rgb(name), markersize=markersize)
            for name in labels
        ]

        return handles, labels

    def to_lut(self) -> np.ndarray:
        """
        Returns an ``(n_types, 3)`` uint8 lookup table indexed by class index, which makes
        colouring hundreds of thousands of cells a single fancy-indexing operation.
        """

        return np.array([self.rgb255(name) for name in self.names], dtype=np.uint8)


def palette_from_yaml(color_dict: Dict[str, list]) -> Palette:
    """
    Rebuilds a palette from a colour dictionary in the ``{"0": ["CellType", [r, g, b, a]]}``
    format, so a run can be re-plotted with the exact colours it was exported with.

    Args:
        color_dict: The colour dictionary, as written by :meth:`Palette.as_geojson_dict`.

    Returns:
        A palette holding those cell types and colours.
    """

    ordered = sorted(color_dict.items(), key=lambda item: int(item[0]))
    palette = Palette([name for _, (name, _) in ordered])
    palette._colors = {name: tuple(c / 255 for c in rgb[:3]) for _, (name, rgb) in ordered}  # type: ignore[misc]

    return palette
