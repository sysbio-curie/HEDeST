"""
The spreadsheet a run leaves behind.

``stats.xlsx`` is written at the end of every run so a run can be judged without opening a
notebook. It holds what can be computed from the predictions alone, which means it needs no
slide, no segmentation and no ground truth:

- ``run`` — the settings and the shape of the run;
- ``cell_types`` — per cell type, how many cells were assigned to it, how confident those
  calls are, and how that compares to the mean deconvolution proportion;
- ``spot_metrics`` — how well the per-spot means reproduce the deconvolution, on all spots
  and on the held-out spots only, which is the honest number;
- ``cell_types_raw`` — the same per-cell-type table before the adjustment, to see what PPSA
  did;
- ``seeds`` — for an aggregate, the per-seed spread and agreement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing import Union

import pandas as pd
from loguru import logger

from hedest.analysis.loaders import HedestRun
from hedest.analysis.pred_analyzer import PredAnalyzer
from hedest.analysis.seeds import SeedEnsemble


def write_stats(
    path: Union[str, Path],
    run: HedestRun,
    ensemble: Optional[SeedEnsemble] = None,
) -> Path:
    """
    Writes the statistics spreadsheet of a run.

    Args:
        path: Destination ``.xlsx`` file.
        run: The run to summarise.
        ensemble: The seeds behind it, when the run is an aggregate, which adds the
            ``seeds`` sheet.

    Returns:
        The path written.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    adjusted = PredAnalyzer(run, adjusted=True)
    sheets = {
        "run": adjusted.describe(),
        "cell_types": adjusted.celltype_summary(),
    }

    try:
        sheets["spot_metrics"] = adjusted.spot_metrics(subset="all")
    except ValueError as err:  # pragma: no cover - only when no spot survives the alignment
        logger.warning(f"Could not compute the spot metrics: {err}")

    if run.train_spot_dict and adjusted.held_out_spots:
        try:
            sheets["spot_metrics_held_out"] = adjusted.spot_metrics(subset="held_out")
        except ValueError as err:
            logger.warning(f"Could not compute the held-out spot metrics: {err}")

    if run.predictions_adjusted is not None:
        sheets["cell_types_raw"] = PredAnalyzer(run, adjusted=False).celltype_summary()

    if ensemble is not None:
        sheets["seeds"] = ensemble.summary()
        sheets["seed_abundance"] = ensemble.abundance_per_seed()

    with pd.ExcelWriter(path) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name)

    logger.info(f"Statistics written to {path} ({', '.join(sheets)})")

    return path


__all__ = ["write_stats"]
