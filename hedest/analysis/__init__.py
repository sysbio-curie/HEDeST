"""
Analysis and visualization tools for HEDeST.

The package has four layers, and you normally only touch the last two:

- :mod:`hedest.analysis.palette` — one colour per cell type, shared by every plot.
- :mod:`hedest.analysis.plots` — small, stateless matplotlib helpers.
- :mod:`hedest.analysis.postseg` — :class:`~hedest.analysis.postseg.SlideVisualizer`, to
  look at a slide with the segmentation, the spots or the predictions drawn on top. Usable
  right after segmentation, before HEDeST has run.
- :mod:`hedest.analysis.pred_analyzer` — :class:`~hedest.analysis.pred_analyzer.PredAnalyzer`,
  everything that needs the predictions of a run.

A finished run is loaded in one call::

    from hedest.analysis import load_run, PredAnalyzer

    run = load_run("results/my_run")                 # or a folder of seed_* runs
    analyzer = PredAnalyzer(run, seg="seg/slide.json", slide_path="slide.tif", mpp=0.2738)

Every plotting function returns a **closed** matplotlib figure. In a notebook, make it the
value of the cell to display it; in a script, pass ``savefig=...`` or call
``fig.savefig(...)``. Nothing displays itself, so nothing is ever drawn twice.

The submodules are imported lazily, so ``import hedest.analysis`` stays cheap and pulling in
one helper does not require the plotly, OpenCV and OpenSlide stack.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "HedestRun",
    "Palette",
    "PredAnalyzer",
    "SeedEnsemble",
    "SlideVisualizer",
    "load_run",
    "load_seed_runs",
    "palette_from_yaml",
    "plots",
    "seeds",
    "stats",
]

_EXPORTS = {
    "HedestRun": ("hedest.analysis.loaders", "HedestRun"),
    "load_run": ("hedest.analysis.loaders", "load_run"),
    "load_seed_runs": ("hedest.analysis.loaders", "load_seed_runs"),
    "Palette": ("hedest.analysis.palette", "Palette"),
    "palette_from_yaml": ("hedest.analysis.palette", "palette_from_yaml"),
    "PredAnalyzer": ("hedest.analysis.pred_analyzer", "PredAnalyzer"),
    "SeedEnsemble": ("hedest.analysis.seeds", "SeedEnsemble"),
    "SlideVisualizer": ("hedest.analysis.postseg", "SlideVisualizer"),
    "plots": ("hedest.analysis", "plots"),
    "seeds": ("hedest.analysis", "seeds"),
    "stats": ("hedest.analysis", "stats"),
}


def __getattr__(name: str) -> Any:
    """Imports an export on first use (PEP 562), keeping the package import light."""

    import importlib

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'hedest.analysis' has no attribute '{name}'") from None

    if module_name == "hedest.analysis":
        return importlib.import_module(f"hedest.analysis.{attribute}")

    return getattr(importlib.import_module(module_name), attribute)


def __dir__() -> list:
    return sorted(__all__)
