"""Scores the two segmentation methods against the three proportion-based ones, at level 0.

HoVerNet and CellViT type a nucleus from its image alone, with the six PanNuke classes and no
spot proportions. At level 0 the STHELAR annotation is three broad categories that the PanNuke
classes map onto one-to-one (``config.PANNUKE_TO_BROAD``), so the two segmentations can be
read as cell-typing methods and scored next to HistoCell, PanoSpace and HEDeST.

Five methods, one annotation level, three classes. For each sample and configuration all five
are scored on **one shared cell set**, the intersection of:

1. cells whose level 0 ground truth is one of the three broad categories;
2. cells predicted by HistoCell, PanoSpace and HEDeST, in all three seeds;
3. cells whose own HoVerNet class maps to a broad category (``nolabe`` and ``necros`` do not);
4. cells matched to a CellViT nucleus whose class maps to a broad category.

Filters 3 and 4 are defined by the two segmentations' own output, so HoVerNet and CellViT are
only scored where they committed to one of the three categories and are never charged for a
cell they called unlabelled or dead. That is worth keeping in mind when reading the figures;
``cell_set_{config}.csv`` records what each filter costs.

Because of the restriction to level 0, to three classes and to this cell set, these numbers
are **not comparable** with those in ``summary_{config}.csv``.

    python benchmark/comparison_code/segmentation_collect.py              # both configurations
    python benchmark/comparison_code/segmentation_collect.py --config gt
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict
from typing import List

import collect as B  # the four-method study: its loaders are reused verbatim
import config as C
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

NO_SEED = -1  # HoVerNet and CellViT are deterministic: one run, no seed


def load_segmenters(sample: str) -> tuple:
    """
    The broad-category prediction of each segmentation, keyed by HoVerNet id.

    Reads the pairs of ``segmentation_match.py``: every row is one nucleus, with the PanNuke
    class HoVerNet and CellViT gave it. Rows where either class has no broad counterpart are
    dropped, which is filters 3 and 4 of the shared cell set.

    Args:
        sample: The sample name.

    Returns:
        ``({"HoVerNet": series, "CellViT": series}, n_pairs)`` on the same index, or
        ``({}, 0)`` when the pairs are missing.
    """

    path = C.seg_pairs_path(sample)
    if not (os.path.exists(path) and os.path.exists(C.seg_meta_path(sample))):
        return {}, 0  # the meta file is written last, so it marks a finished matching

    pairs = pd.read_csv(path, dtype={"hovernet_id": str, "cellvit_id": str})
    broad = {
        "HoVerNet": pairs["hovernet_type"].map(C.PANNUKE_TO_BROAD),
        "CellViT": pairs["cellvit_type"].map(C.PANNUKE_TO_BROAD),
    }
    keep = broad["HoVerNet"].notna() & broad["CellViT"].notna()
    index = pairs.loc[keep, "hovernet_id"].to_numpy()

    series = {method: pd.Series(values[keep].to_numpy(), index=index, dtype=object) for method, values in broad.items()}

    return series, len(pairs)


def load_meta(sample: str) -> dict:
    """
    The nucleus counts and PanNuke histograms written beside the pairs.

    Args:
        sample: The sample name.

    Returns:
        The meta dictionary, or ``{}`` when it is missing.
    """

    path = C.seg_meta_path(sample)
    if not os.path.exists(path):
        return {}

    with open(path) as handle:
        return json.load(handle)


def _mapped(histogram: Dict[str, int]) -> int:
    """How many cells of a PanNuke histogram have a broad counterpart."""

    return int(sum(n for name, n in histogram.items() if name in C.PANNUKE_TO_BROAD))


def score_sample(sample: str, config: str) -> tuple:
    """
    Scores the five methods of one sample on a single shared cell set.

    Args:
        sample: The sample name.
        config: ``"gt"`` or ``"deconv"``.

    Returns:
        ``(metric rows, confusion rows, cell-set row)``; all empty when something is missing.
    """

    level = C.SEG_LEVEL
    segmenters, n_matched = load_segmenters(sample)
    if not segmenters:
        print(f"   [skip] {sample} ({config}): no pairs; run segmentation_match.py first", flush=True)
        return [], [], {}

    # Only the winning type of each cell is kept, as in the four-method study.
    predictions: Dict[tuple, pd.Series] = {(method, NO_SEED): series for method, series in segmenters.items()}
    for seed in C.SEEDS:
        for method in ("HistoCell", "PanoSpace"):
            table = B.load_competitor(method, sample, level, config, seed)
            if table is not None:
                predictions[(method, seed)] = B.argmax_labels(table)
        hedest = B.load_hedest(sample, level, config, seed)
        if "HEDeST" in hedest:  # the raw output; PPSA is not part of this study
            predictions[("HEDeST", seed)] = B.argmax_labels(hedest["HEDeST"])

    missing = [m for m in C.SEG_METHODS if not any(key[0] == m for key in predictions)]
    if missing:
        print(f"   [skip] {sample} ({config}): no prediction for {missing}", flush=True)
        return [], [], {}

    truth = B.load_truth(sample, level)
    truth = truth[truth.isin(C.BROAD_TYPES)]

    shared = set(truth.index)
    for series in predictions.values():
        shared &= set(series.index)
    shared = sorted(shared)
    if not shared:
        print(f"   [skip] {sample} ({config}): no cell shared by every method", flush=True)
        return [], [], {}

    y_true = truth.loc[shared].to_numpy()
    present = [t for t in C.BROAD_TYPES if t in set(y_true)]  # lymph_node_s0 has no Epithelial

    metrics, confusions = [], []
    for (method, seed), series in predictions.items():
        y_pred = series.loc[shared].to_numpy()
        metrics.append(
            {
                "config": config,
                "sample": sample,
                "level": level,
                "method": method,
                "seed": seed,
                "n_true_types": len(present),
                "n_cells": len(shared),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        )
        counts = confusion_matrix(y_true, y_pred, labels=C.BROAD_TYPES)
        for i, true_type in enumerate(C.BROAD_TYPES):
            for j, pred_type in enumerate(C.BROAD_TYPES):
                confusions.append(
                    {
                        "config": config,
                        "sample": sample,
                        "method": method,
                        "seed": seed,
                        "true": true_type,
                        "pred": pred_type,
                        "n": int(counts[i, j]),
                    }
                )

    # What each filter of the shared cell set costs, from all the nuclei down to the scored ones.
    meta = load_meta(sample)
    cell_set = {
        "config": config,
        "sample": sample,
        "n_hovernet": meta.get("n_hovernet"),
        "n_hovernet_broad": _mapped(meta.get("hovernet_types", {})),
        "n_cellvit": meta.get("n_cellvit"),
        "n_cellvit_broad": _mapped(meta.get("cellvit_types", {})),
        "n_matched": n_matched,
        "n_matched_broad": len(next(iter(segmenters.values()))),
        "n_with_gt": len(truth),
        "n_scored": len(shared),
        "n_true_types": len(present),
    }
    for name in C.DROPPED_PANNUKE:
        cell_set[f"n_hovernet_{name}"] = int(meta.get("hovernet_types", {}).get(name, 0))
        cell_set[f"n_cellvit_{name}"] = int(meta.get("cellvit_types", {}).get(name, 0))

    best = max(metrics, key=lambda row: row["balanced_accuracy"])
    print(
        f"   {sample} ({config}): {len(shared)} scored cells, {len(present)} types, "
        f"best {best['method']} {best['balanced_accuracy']:.3f}",
        flush=True,
    )

    return metrics, confusions, cell_set


def _score_one(task: tuple) -> tuple:
    """Worker wrapper: ``score_sample`` over a ``(sample, config)`` tuple."""

    return score_sample(*task)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and 95% CI of the balanced accuracy over the seeds, per sample and method.

    HoVerNet and CellViT have a single run, so their CI is 0 and ``n_seeds`` is 1; the three
    proportion-based methods average their three seeds.

    Args:
        metrics: The per-seed table.

    Returns:
        One row per (sample, method).
    """

    keys = ["config", "sample", "level", "n_true_types", "n_cells", "method"]
    grouped = metrics.groupby(keys, sort=False)["balanced_accuracy"]
    summary = pd.DataFrame(
        {"mean": grouped.mean(), "ci": grouped.apply(B.ci95), "n_seeds": grouped.size()}
    ).reset_index()
    order = {method: i for i, method in enumerate(C.SEG_METHODS)}

    return summary.sort_values(["sample", "method"], key=lambda col: col.map(order) if col.name == "method" else col)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=list(C.CONFIGS) + ["both"], default="both", help="Which configuration.")
    parser.add_argument("--out-dir", default=C.SEG_DIR, help="Where the tables go.")
    parser.add_argument("--workers", type=int, default=4, help="Samples scored in parallel (1 = serial).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = list(C.CONFIGS) if args.config == "both" else [args.config]

    for config in configs:
        print(f"== {config}: {C.CONFIGS[config]}, {C.SEG_LEVEL}, {len(C.BROAD_TYPES)} broad types", flush=True)
        todo = C.seg_units(config)
        metrics: List[dict] = []
        confusions: List[dict] = []
        cell_sets: List[dict] = []

        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                parts = pool.map(_score_one, [(s, config) for s in todo])
        else:
            parts = (score_sample(s, config) for s in todo)

        for part_metrics, part_confusions, part_cell_set in parts:
            metrics.extend(part_metrics)
            confusions.extend(part_confusions)
            if part_cell_set:
                cell_sets.append(part_cell_set)

        if not metrics:
            print(f"   nothing scored for {config}", flush=True)
            continue

        table = pd.DataFrame(metrics)
        table.to_csv(os.path.join(args.out_dir, f"metrics_{config}.csv"), index=False)
        summary = summarise(table)
        summary.to_csv(os.path.join(args.out_dir, f"summary_{config}.csv"), index=False)
        pd.DataFrame(confusions).to_csv(os.path.join(args.out_dir, f"confusion_{config}.csv"), index=False)
        pd.DataFrame(cell_sets).to_csv(os.path.join(args.out_dir, f"cell_set_{config}.csv"), index=False)

        wide = summary.pivot_table(index="sample", columns="method", values="mean")
        wide = wide[[m for m in C.SEG_METHODS if m in wide.columns]]
        n_cells = int(summary.drop_duplicates("sample")["n_cells"].sum())
        print(f"\n{len(table)} rows, {summary['sample'].nunique()} samples, {n_cells} scored cells")
        print(wide.round(3).to_string(), "\n")
        print("mean over samples:")
        print(wide.mean().round(4).to_string(), "\n")

        # lymph_node_s0 has no Epithelial at level 0, so the three proportion-based methods
        # cannot even emit that class there while the two segmentations can (and are wrong
        # whenever they do). The mean over the three-class samples only is reported beside it.
        three = summary.drop_duplicates("sample").set_index("sample")["n_true_types"] == len(C.BROAD_TYPES)
        if not three.all():
            print(f"mean over the {int(three.sum())} three-class samples ({', '.join(three[~three].index)} out):")
            print(wide.loc[three[three].index].mean().round(4).to_string(), "\n")


if __name__ == "__main__":
    main()
