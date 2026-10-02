"""
Combining the seeds of one HEDeST study.

HEDeST is supervised by spot-level proportions only, so different seeds can assign a given
cell differently while fitting the spots equally well. Averaging a handful of seeds gives a
better prediction, and the spread between them is a confidence measure that costs nothing
extra.

This script reads every ``seed_*/info.pickle`` of a directory and writes, next to them:

- ``info_aggregated.pickle`` — the mean predictions, plus the per-cell standard deviation
  and the per-cell seed agreement, loadable with ``load_run``;
- ``stats_aggregated.xlsx`` — the usual statistics, with two extra sheets on the seeds;
- ``hedest_predictions_aggregated*.geojson`` — the aggregated labels for QuPath, when the
  segmentation is given.

Usage::

    python hedest/aggregate_seeds.py <run_dir> [--json-path seg.json] [--color-dict-file colors.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from loguru import logger

from hedest.analysis.loaders import AGGREGATED_NAME
from hedest.analysis.loaders import HedestRun
from hedest.analysis.loaders import load_seed_runs
from hedest.analysis.palette import Palette
from hedest.analysis.palette import palette_from_yaml
from hedest.analysis.pred_analyzer import PredAnalyzer
from hedest.analysis.seeds import SeedEnsemble
from hedest.analysis.stats import write_stats


def aggregate_seeds(
    run_dir: Path,
    json_path: Optional[str] = None,
    color_dict_file: Optional[str] = None,
) -> Optional[HedestRun]:
    """
    Aggregates every seed run of a directory.

    Args:
        run_dir: Directory holding the ``seed_*`` subdirectories.
        json_path: Path to the segmentation JSON, to also export the GeoJSON files.
        color_dict_file: YAML colour dictionary to keep the colours of an earlier export.

    Returns:
        The aggregated run, or None when no seed was found.
    """

    run_dir = Path(run_dir)
    runs = load_seed_runs(run_dir)

    if not runs:
        logger.error(f"No seed_*/info.pickle found under {run_dir}. Nothing to aggregate.")
        return None

    logger.info(f"Aggregating {len(runs)} seeds: {[run.run_dir.name for run in runs if run.run_dir]}")
    ensemble = SeedEnsemble(runs)
    aggregated = ensemble.to_run(run_dir=run_dir)
    aggregated.save(run_dir / AGGREGATED_NAME)

    agreement = ensemble.agreement()
    logger.info(
        f"-> mean seed agreement {agreement.mean():.3f}, unanimous on "
        f"{float((agreement == 1.0).mean()):.1%} of the {len(agreement)} cells."
    )

    write_stats(run_dir / "stats_aggregated.xlsx", aggregated, ensemble=ensemble)

    if json_path is not None:
        if color_dict_file is not None:
            import yaml

            with open(color_dict_file) as color_file:
                palette = palette_from_yaml(yaml.safe_load(color_file))
        else:
            palette = Palette(aggregated.ct_list)

        for adjusted, name in (
            (True, "hedest_predictions_aggregated_adj.geojson"),
            (False, "hedest_predictions_aggregated.geojson"),
        ):
            analyzer = PredAnalyzer(aggregated, seg=json_path, palette=palette, adjusted=adjusted)
            analyzer.export_geojson(run_dir / name)

    logger.info(f"Aggregation finished in {run_dir}")

    return aggregated


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description="Aggregate the seed runs of a HEDeST study.")
    parser.add_argument("run_dir", type=Path, help="Directory holding the seed_* subdirectories.")
    parser.add_argument("--json-path", default=None, help="Segmentation JSON, to export the GeoJSON files too.")
    parser.add_argument("--color-dict-file", default=None, help="YAML colour dictionary (special format).")
    arguments = parser.parse_args()

    aggregate_seeds(
        run_dir=arguments.run_dir,
        json_path=arguments.json_path,
        color_dict_file=arguments.color_dict_file,
    )


if __name__ == "__main__":
    main()
