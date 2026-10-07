"""Step 5 - score and plot the HEDeST runs of the fully simulated datasets.

Reads the runs written by ``04_hedest.py`` under ``models/full_sim/{sample}/seed_*/`` and
writes, in ``plots/hedest_results/`` (next to the ``construction/`` and ``datasets/``
figures of steps 1 to 3):

  results_per_seed.csv              one row per (sample, variant, seed): the cell-level metrics
  results_summary.csv               mean and 95% CI of the balanced accuracy over the seeds
  ppsa_all_samples                  every sample, without and with PPSA, chance level marked
  perturbation                      balanced accuracy against the perturbation strength
  k_series                          balanced accuracy against the number of clusters
  design_6clusters                  base vs dup vs dup_not_mixed
  confusion/{sample}                row-normalised confusion, no PPSA | PPSA | difference
  galleries/{sample}_pred.png       crops grouped by predicted cluster (most confident cells)
  galleries/{sample}_gt.png         the same cells grouped by true cluster

"variant" is the model output (``raw``) or the PPSA-adjusted one (``ppsa``); PPSA is applied
inside spots only, which is all these datasets have. The curves come from the 10 seeds, the
confusion matrices and the galleries from their mean probabilities. Figures are png + svg,
except the galleries (bitmaps, png only). The mean probabilities of each sample are also
written next to the runs as ``predictions_{variant}.csv.gz``.

    python simulations/full_simulations/05_analyse.py
    python simulations/full_simulations/05_analyse.py --only 30spots --no-galleries
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from typing import Dict
from typing import List

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import config as C  # noqa: E402
from utils import analysis as A  # noqa: E402


def _load_runner():
    """Imports ``04_hedest.py``, whose name is not a valid module name, for its sample list."""

    spec = importlib.util.spec_from_file_location("hedest_runner", os.path.join(HERE, "04_hedest.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


_runner = _load_runner()
PLOT_DIR = C.RESULT_PLOT_DIR
OUT_ROOT = _runner.OUT_ROOT


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def specs() -> List[dict]:
    """
    Every sample with the design metadata the figures group by.

    Returns:
        The samples of ``04_hedest.samples`` enriched with ``K``, ``balance``, ``n_spots``,
        ``dup``, ``not_mixed`` and a short ``design`` label.
    """

    by_tag = {C.dataset_tag(d): d for d in C.DATASETS}
    out = []
    for sample in _runner.samples():
        d = by_tag[sample["tag"]]
        design = "base"
        if d["dup"]:
            design = "dup_not_mixed" if d["not_mixed"] else "dup"
        out.append(
            {
                **sample,
                "K": d["K"],
                "balance": d["balance"],
                "n_spots": d["n_spots"],
                "dup": bool(d["dup"]),
                "not_mixed": bool(d["not_mixed"]),
                "design": design,
                "n_clusters": d["K"] + int(d["dup"]),
            }
        )

    return out


def collect(todo: List[dict]) -> pd.DataFrame:
    """
    Scores every seed of every sample against the ground truth.

    Args:
        todo: The samples to score.

    Returns:
        One row per (sample, variant, seed).
    """

    rows = []
    for n, sample in enumerate(todo, 1):
        sample_dir = os.path.join(OUT_ROOT, sample["name"])
        if not os.path.isdir(sample_dir):
            log(f"[{n}/{len(todo)}] {sample['name']}: no run, skipped")
            continue
        gt = A.load_gt(C.SIM_DIR, sample["tag"])
        table = A.seed_metrics(sample_dir, gt)
        for key, value in sample.items():
            if key != "prop":
                table[key] = value
        rows.append(table)
        log(
            f"[{n}/{len(todo)}] {sample['name']}: {table['seed'].nunique()} seeds, "
            f"bal. acc. {table[table.variant == 'raw'][A.BA].mean():.3f} raw / "
            f"{table[table.variant == 'ppsa'][A.BA].mean():.3f} PPSA"
        )

    if not rows:
        raise SystemExit(f"No run found under {OUT_ROOT}. Run 04_hedest.py first.")

    results = pd.concat(rows, ignore_index=True)
    columns = ["name", "tag", "K", "n_clusters", "balance", "n_spots", "design", "strength", "variant", "seed"]
    rest = [c for c in results.columns if c not in columns]

    return results[columns + rest].rename(columns={"name": "sample"})


def figures(results: pd.DataFrame, todo: List[dict], galleries: bool = True, per_sample: bool = True) -> None:
    """
    Writes every figure of the study.

    Args:
        results: The per-seed table of :func:`collect`.
        todo: The samples, for the per-sample figures.
        galleries: Whether to draw the crop galleries (the slow part).
        per_sample: Whether to draw the per-sample confusion matrices and galleries at all.
    """

    chance = results.groupby("sample")["n_clusters"].first().rdiv(1.0)
    A.plot_ppsa_bars(
        A.summarise(results, ["sample", "variant"]),
        os.path.join(PLOT_DIR, "ppsa_all_samples"),
        "HEDeST on the fully simulated datasets (10 seeds, 95% CI)",
        chance=chance,
    )

    perturbed = results[results["tag"].isin(results.loc[results["strength"] > 0, "tag"].unique())]
    if not perturbed.empty:
        A.plot_perturbation(
            A.summarise(perturbed, ["tag", "strength", "variant"]),
            os.path.join(PLOT_DIR, "perturbation"),
            "Effect of perturbing the spot proportions (10 seeds, 95% CI)",
        )

    series = results[(results["n_spots"] == 200) & (~results["dup"]) & (results["strength"] == 0)]
    if not series.empty:
        A.plot_k_series(
            A.summarise(series, ["K", "balance", "variant"]),
            os.path.join(PLOT_DIR, "k_series"),
            "Effect of the number of clusters (200 spots, 10 seeds, 95% CI)",
        )

    family = results[(results["K"] == 6) & (results["strength"] == 0)]
    if not family.empty:
        A.plot_design_bars(
            A.summarise(family, ["design", "balance", "variant"]),
            os.path.join(PLOT_DIR, "design_6clusters"),
            "Duplicated cluster, mixed or not (6 clusters, 200 spots, 10 seeds, 95% CI)",
            order=["base", "dup", "dup_not_mixed"],
        )

    if not per_sample:
        return

    cache: Dict[str, dict] = {}
    for n, sample in enumerate(todo, 1):
        sample_dir = os.path.join(OUT_ROOT, sample["name"])
        if not os.path.isdir(sample_dir):
            continue
        tag = sample["tag"]
        if tag not in cache:
            cache.clear()
            cache[tag] = {
                "gt": A.load_gt(C.SIM_DIR, tag),
                "images": A.load_images(C.SIM_DIR, tag) if galleries else None,
            }
        gt, images = cache[tag]["gt"], cache[tag]["images"]

        matrices, scores = {}, {}
        for variant in A.VARIANTS:
            an = A.analyzer(sample_dir, gt, variant, image_dict=images)
            metrics = an.cell_metrics(per_class=True)
            matrices[variant] = metrics["confusion_matrix"]
            scores[variant] = metrics["balanced_accuracy"]
            an.run.predictions.to_csv(os.path.join(sample_dir, f"predictions_{variant}.csv.gz"))
            if galleries and variant == "ppsa":
                A.plot_galleries(an, gt, os.path.join(PLOT_DIR, "galleries", sample["name"]), sample["name"])

        A.plot_confusion(
            matrices,
            os.path.join(PLOT_DIR, "confusion", sample["name"]),
            f"{sample['name']} — mean of 10 seeds",
            scores=scores,
        )
        log(f"[{n}/{len(todo)}] {sample['name']}: figures written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=str, default=None, help="Keep only the samples whose name contains this.")
    parser.add_argument("--no-galleries", action="store_true", help="Skip the crop galleries.")
    parser.add_argument("--no-figures", action="store_true", help="Only write the tables.")
    parser.add_argument(
        "--summary-only", action="store_true", help="Only the overview figures, not the per-sample ones."
    )
    parser.add_argument("--plot-dir", type=str, default=PLOT_DIR, help="Where the figures go.")
    args = parser.parse_args()

    globals()["PLOT_DIR"] = args.plot_dir
    os.makedirs(args.plot_dir, exist_ok=True)

    todo = specs()
    if args.only:
        todo = [s for s in todo if args.only in s["name"]]
    log(f"{len(todo)} samples from {OUT_ROOT} -> {args.plot_dir}")

    start = time.time()
    results = collect(todo)
    results.to_csv(os.path.join(args.plot_dir, "results_per_seed.csv"), index=False)

    keys = ["sample", "tag", "K", "n_clusters", "balance", "n_spots", "design", "strength", "variant"]
    summary = A.summarise(results, keys)
    summary.to_csv(os.path.join(args.plot_dir, "results_summary.csv"), index=False)
    log(f"tables written ({len(results)} rows, {len(summary)} groups)")

    if not args.no_figures:
        figures(results, todo, galleries=not args.no_galleries, per_sample=not args.summary_only)

    log(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
