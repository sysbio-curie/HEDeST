"""Scores HistoCell, PanoSpace, HEDeST and HEDeST + PPSA against the matched DAPI ground truth.

For one sample-level and one configuration, the four methods are scored on **exactly the same
cells**: the intersection of the cells every method predicted, over all seeds, with the cells
that have a ground-truth type among the proportion columns. PanoSpace in particular does not
cover every nucleus, so without this the methods would be scored on different populations and
their balanced accuracies would not be comparable.

Per-cell predictions come from
``{HistoCell,PanoSpace}/{sample}/{level}/{gt,endecon}/seed{seed}/*_predictions.csv`` and, for
both HEDeST variants, from the ``predictions.npz`` of one run (``raw`` and ``ppsa`` are the
same model before and after the adjustment).

    python benchmark/comparison_code/collect.py              # both configurations
    python benchmark/comparison_code/collect.py --config gt
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict
from typing import List
from typing import Optional

import config as C
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

PRED_FILE = {"HistoCell": "histocell_predictions.csv", "PanoSpace": "panospace_predictions.csv"}


def argmax_labels(table: pd.DataFrame) -> pd.Series:
    """
    The most probable cell type of every cell.

    ``DataFrame.idxmax`` is slow on these matrices (up to 632k rows), so the argmax is taken
    on the numpy array and mapped back to the column names.

    Args:
        table: A ``cells x types`` probability matrix.

    Returns:
        ``{cell id: cell type}``.
    """

    columns = np.asarray(table.columns, dtype=object)

    return pd.Series(columns[table.to_numpy().argmax(axis=1)], index=table.index)


def load_competitor(method: str, sample: str, level: str, config: str, seed: int) -> Optional[pd.DataFrame]:
    """
    The per-cell probabilities of HistoCell or PanoSpace.

    Args:
        method: ``"HistoCell"`` or ``"PanoSpace"``.
        sample: The sample name.
        level: The annotation level.
        config: ``"gt"`` or ``"deconv"``.
        seed: The seed.

    Returns:
        A ``cells x types`` frame indexed by cell id as a string, or None when absent.
    """

    path = os.path.join(C.competitor_dir(method, sample, level, config, seed), PRED_FILE[method])
    if not os.path.exists(path):
        return None

    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str)

    return table


def load_hedest(sample: str, level: str, config: str, seed: int) -> Dict[str, pd.DataFrame]:
    """
    The per-cell probabilities of both HEDeST variants of one run.

    Args:
        sample: The sample name.
        level: The annotation level.
        config: ``"gt"`` or ``"deconv"``.
        seed: The seed.

    Returns:
        ``{"HEDeST": frame, "HEDeST + PPSA": frame}``, empty when the run is missing.
    """

    run_dir = C.hedest_run_dir(sample, level, config, seed)
    npz_path = os.path.join(run_dir, "predictions.npz")
    ids_path = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "cell_ids.txt")
    if not (os.path.exists(npz_path) and os.path.exists(ids_path)):
        return {}

    with open(ids_path) as handle:
        cell_ids = [line.strip() for line in handle if line.strip()]
    data = np.load(npz_path, allow_pickle=True)
    classes = [str(c) for c in data["classes"]]

    out = {}
    for method, key in C.HEDEST_METHODS.items():
        out[method] = pd.DataFrame(data[key].astype(np.float32), index=cell_ids, columns=classes)

    return out


def load_truth(sample: str, level: str) -> pd.Series:
    """
    The matched DAPI ground truth of a sample-level.

    Args:
        sample: The sample name.
        level: The annotation level.

    Returns:
        ``{cell id: cell type}`` as a Series.
    """

    with open(C.gt_path(sample, level)) as handle:
        gt = json.load(handle)["gt"]

    return pd.Series(gt, dtype=object)


def score_unit(sample: str, level: str, config: str) -> List[dict]:
    """
    Scores every method and seed of one sample-level on a single shared cell set.

    Args:
        sample: The sample name.
        level: The annotation level.
        config: ``"gt"`` or ``"deconv"``.

    Returns:
        One row per (method, seed); empty when something is missing.
    """

    # Only the winning type of each cell is kept: the probability matrices are large and
    # nothing downstream needs them.
    predictions: Dict[tuple, pd.Series] = {}
    classes: set = set()
    for seed in C.SEEDS:
        for method in PRED_FILE:
            table = load_competitor(method, sample, level, config, seed)
            if table is not None:
                classes |= set(table.columns)
                predictions[(method, seed)] = argmax_labels(table)
        for method, table in load_hedest(sample, level, config, seed).items():
            classes |= set(table.columns)
            predictions[(method, seed)] = argmax_labels(table)

    missing = [m for m in C.METHODS if not any(k[0] == m for k in predictions)]
    if missing:
        print(f"   [skip] {sample}/{level} ({config}): no prediction for {missing}", flush=True)
        return []

    truth = load_truth(sample, level)
    classes = sorted(classes)
    truth = truth[truth.isin(classes)]

    shared = set(truth.index)
    for series in predictions.values():
        shared &= set(series.index)
    shared = sorted(shared)
    if not shared:
        print(f"   [skip] {sample}/{level} ({config}): no cell shared by every method", flush=True)
        return []

    y_true = truth.loc[shared].to_numpy()
    rows = []
    for (method, seed), series in sorted(predictions.items()):
        y_pred = series.loc[shared].to_numpy()
        rows.append(
            {
                "config": config,
                "sample": sample,
                "level": level,
                "level_num": int(level.removeprefix("level")),
                "method": method,
                "seed": seed,
                "n_types": len(classes),
                "n_cells": len(shared),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        )

    best = max(rows, key=lambda r: r["balanced_accuracy"])
    print(
        f"   {sample}/{level} ({config}): {len(shared)} shared cells, {len(classes)} types, "
        f"best {best['method']} {best['balanced_accuracy']:.3f}",
        flush=True,
    )

    return rows


def _score_one(task: tuple) -> List[dict]:
    """Worker wrapper: ``score_unit`` over a ``(sample, level, config)`` tuple."""

    return score_unit(*task)


def ci95(values: pd.Series) -> float:
    """
    Half-width of the t-based 95% confidence interval of the mean (0 with fewer than 2 values).

    Args:
        values: The values.

    Returns:
        The half-width.
    """

    from scipy import stats

    n = values.count()
    if n < 2:
        return 0.0

    return float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=list(C.CONFIGS) + ["both"], default="both", help="Which configuration.")
    parser.add_argument("--out-dir", default=C.RESULT_DIR, help="Where the tables go.")
    parser.add_argument("--workers", type=int, default=6, help="Sample-levels scored in parallel (1 = serial).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = list(C.CONFIGS) if args.config == "both" else [args.config]

    for config in configs:
        print(f"== {config}: {C.CONFIGS[config]}", flush=True)
        todo = C.units(config)
        rows = []
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for part in pool.map(_score_one, [(s, lvl, config) for s, lvl in todo]):
                    rows.extend(part)
        else:
            for sample, level in todo:
                rows.extend(score_unit(sample, level, config))
        if not rows:
            print(f"   nothing scored for {config}", flush=True)
            continue

        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(args.out_dir, f"metrics_{config}.csv"), index=False)

        keys = ["config", "sample", "level", "level_num", "n_types", "n_cells", "method"]
        grouped = table.groupby(keys, sort=False)["balanced_accuracy"]
        summary = pd.DataFrame(
            {"mean": grouped.mean(), "ci": grouped.apply(ci95), "n_seeds": grouped.size()}
        ).reset_index()
        summary.to_csv(os.path.join(args.out_dir, f"summary_{config}.csv"), index=False)

        wide = summary.pivot_table(index=["sample", "level"], columns="method", values="mean")
        print(f"\n{len(table)} rows, {table[['sample', 'level']].drop_duplicates().shape[0]} sample-levels")
        print(wide.round(3).to_string(), "\n")
        print("mean over sample-levels:")
        print(wide.mean().round(4).to_string(), "\n")


if __name__ == "__main__":
    main()
