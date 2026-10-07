"""Random vs MHAST vs HEDeST on the 30-spot fully simulated datasets.

The four methods compared on the same cells, the same spots and the same 10 seeds:

* **Random** — the initial permutation MHAST starts from: inside every spot the true labels
  are shuffled, so the spot composition is exact and only the assignment is chance. This is
  the "before" state ``external/mhast/sim.py`` already measures.
* **MHAST** — the hierarchical permutation of that initial state (its "after" state),
  run by ``external/mhast/run_sim.sh``, one seed per array task.
* **HEDeST** and **HEDeST + PPSA** — the runs of ``simulations/full_simulations/04_hedest.py``,
  scored here with ``hedest.analysis.PredAnalyzer`` so that every number in the figure comes
  from one definition of the metrics.

Only the 30-spot datasets are compared: MHAST's global stage enumerates a Cartesian product
(about 1e6 candidates on 136 cells) and does not scale to the 200-spot designs.

Writes, under ``benchmark/results/MHAST/``:

    comparison_per_seed.csv        one row per (dataset, method, seed)
    comparison_summary.csv         mean and 95% CI per (dataset, method)
    {tag}_metrics.xlsx             one summary sheet per method, plus the per-seed sheets
    plots/balanced_accuracy_30spots.png|svg     the comparison asked for
    plots/{tag}_all_metrics.png                 the four metrics, via benchmark.utils

    python benchmark/mhast_benchmark.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from benchmark.utils import bar_plot_perf  # noqa: E402
from benchmark.utils import compute_statistics  # noqa: E402
from hedest.analysis import load_seed_runs  # noqa: E402
from hedest.analysis import PredAnalyzer  # noqa: E402

SIM_DIR = "/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/sim"
MHAST_DIR = os.path.join(REPO, "benchmark", "results", "MHAST")
HEDEST_ROOT = os.path.join(REPO, "models", "full_sim")
TAGS = [
    "4_hoptimus_clusters_30spots_balanced_5mean_5var",
    "4_hoptimus_clusters_30spots_imbalanced_5mean_5var",
]

# MHAST's metric names are the reference: benchmark.utils reads the spreadsheets by them.
METRICS = {
    "accuracy": "Global Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "weighted_f1": "Weighted F1 Score",
    "weighted_precision": "Weighted Precision",
    "weighted_recall": "Weighted Recall",
}
METHODS = {
    "Random": "#bdbdbd",
    "MHAST": "#fd8d3c",
    "HEDeST": "#9ecae1",
    "HEDeST + PPSA": "#08519c",
}


def mhast_metrics(tag: str, mhast_dir: str) -> pd.DataFrame:
    """
    The per-seed metrics of MHAST and of the random permutation it starts from.

    Args:
        tag: The dataset tag.
        mhast_dir: ``benchmark/results/MHAST``, holding ``{tag}/seed_*.xlsx``.

    Returns:
        One row per (method, seed), with MHAST's metric names.

    Raises:
        FileNotFoundError: If no seed spreadsheet is there.
    """

    folder = os.path.join(mhast_dir, tag)
    files = (
        sorted(
            (int(os.path.basename(p).removeprefix("seed_").removesuffix(".xlsx")), os.path.join(folder, p))
            for p in os.listdir(folder)
            if p.startswith("seed_") and p.endswith(".xlsx")
        )
        if os.path.isdir(folder)
        else []
    )

    if not files:
        raise FileNotFoundError(f"No seed_*.xlsx under {folder}. Run: sbatch external/mhast/run_sim.sh")

    rows = []
    for seed, path in files:
        for sheet, method in (("Metrics Before", "Random"), ("Metrics After", "MHAST")):
            table = pd.read_excel(path, sheet_name=sheet)
            for _, run in table.iterrows():
                rows.append({"method": method, "seed": seed, **{m: run[m] for m in METRICS.values()}})

    return pd.DataFrame(rows)


def hedest_metrics(tag: str, hedest_root: str, sim_dir: str) -> pd.DataFrame:
    """
    The per-seed metrics of the HEDeST runs, with and without PPSA.

    Args:
        tag: The dataset tag, which is also the sample folder name.
        hedest_root: ``models/full_sim``.
        sim_dir: The dataset folder, for the ground truth.

    Returns:
        One row per (method, seed), with MHAST's metric names.

    Raises:
        FileNotFoundError: If no seed run is there.
    """

    gt = pd.read_csv(os.path.join(sim_dir, f"{tag}_gt.csv"), index_col=0)
    gt.index = gt.index.astype(str)

    rows = []
    for method, adjusted in (("HEDeST", False), ("HEDeST + PPSA", True)):
        runs = load_seed_runs(os.path.join(hedest_root, tag), adjusted=adjusted)
        if not runs:
            raise FileNotFoundError(
                f"No seed run under {os.path.join(hedest_root, tag)}. "
                "Run: sbatch simulations/full_simulations/run_hedest_sim.sh"
            )
        for run in runs:
            # PredAnalyzer defaults to adjusted=True and would switch a raw run back.
            metrics = PredAnalyzer(run, ground_truth=gt, adjusted=adjusted).cell_metrics(per_class=False)
            rows.append(
                {"method": method, "seed": run.seeds[0], **{long: metrics[short] for short, long in METRICS.items()}}
            )

    return pd.DataFrame(rows)


def write_spreadsheet(per_seed: pd.DataFrame, path: str) -> None:
    """
    One spreadsheet per dataset: a summary sheet per method, then its per-seed rows.

    The summary sheets hold ``{metric}`` and ``{metric} ci`` on a single row, which is the
    layout :func:`benchmark.utils.bar_plot_perf` reads.

    Args:
        per_seed: The per-seed table of one dataset.
        path: Destination ``.xlsx``.
    """

    columns = list(METRICS.values())
    with pd.ExcelWriter(path) as writer:
        for method in METHODS:
            sub = per_seed[per_seed["method"] == method]
            if sub.empty:
                continue
            mean, ci = compute_statistics(sub[columns].to_dict(orient="records"))
            pd.DataFrame([{**mean, **ci}]).to_excel(writer, sheet_name=method.replace(" + ", "_"), index=False)
            sub.to_excel(writer, sheet_name=f"{method.replace(' + ', '_')} runs"[:31], index=False)


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


def plot_balanced_accuracy(per_seed: pd.DataFrame, stem: str) -> None:
    """
    Balanced accuracy of the four methods on the two 30-spot datasets.

    Args:
        per_seed: The per-seed table of every dataset.
        stem: Path without extension; ``png`` and ``svg`` are written.
    """

    metric = METRICS["balanced_accuracy"]
    summary = per_seed.groupby(["dataset", "method"], sort=False)[metric].agg(["mean", ci95, "size"])
    summary.columns = ["mean", "ci", "n_seeds"]
    summary = summary.reset_index()

    datasets = [t for t in TAGS if t in set(summary["dataset"])]
    methods = [m for m in METHODS if m in set(summary["method"])]
    x = np.arange(len(datasets))
    width = 0.78 / max(len(methods), 1)

    # benchmark.utils sets a seaborn style globally; this figure keeps the matplotlib defaults.
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(3.6 * len(datasets) + 3.0, 5.2))
        for i, method in enumerate(methods):
            sub = summary[summary["method"] == method].set_index("dataset").reindex(datasets)
            offset = (i - (len(methods) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                sub["mean"],
                width,
                yerr=sub["ci"],
                label=method,
                color=METHODS[method],
                error_kw={"elinewidth": 1.2, "ecolor": "0.25", "capsize": 3},
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=7)

        chance = 1.0 / 4
        ax.axhline(chance, color="#d94801", linestyle="--", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [t.replace("_hoptimus_clusters_", " clusters, ").replace("_5mean_5var", "") for t in datasets]
        )
        ax.set_ylabel("balanced accuracy")
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(
            f"Cell-level balanced accuracy on the 30-spot fully simulated datasets "
            f"({int(summary['n_seeds'].max())} seeds, 95% CI)",
            pad=30,
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color="#d94801", linestyle="--", linewidth=1.2))
        labels.append(f"chance (1/4 = {chance:.2f})")
        ax.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(labels))

        os.makedirs(os.path.dirname(stem), exist_ok=True)
        for ext in ("png", "svg"):
            fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mhast-dir", default=MHAST_DIR, help="Where run_sim.sh wrote the MHAST spreadsheets.")
    parser.add_argument("--hedest-root", default=HEDEST_ROOT, help="Where 04_hedest.py wrote the runs.")
    parser.add_argument("--sim-dir", default=SIM_DIR, help="The simulated datasets, for the ground truth.")
    parser.add_argument("--out-dir", default=MHAST_DIR, help="Where the tables and the plots go.")
    args = parser.parse_args()

    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    tables: List[pd.DataFrame] = []
    for tag in TAGS:
        per_seed = pd.concat(
            [mhast_metrics(tag, args.mhast_dir), hedest_metrics(tag, args.hedest_root, args.sim_dir)],
            ignore_index=True,
        )
        per_seed.insert(0, "dataset", tag)
        tables.append(per_seed)

        spreadsheet = os.path.join(args.out_dir, f"{tag}_metrics.xlsx")
        write_spreadsheet(per_seed, spreadsheet)
        print(f"-> {spreadsheet}")

        sheets = [
            (spreadsheet, method.replace(" + ", "_"), method, color)
            for method, color in METHODS.items()
            if method in set(per_seed["method"])
        ]
        bar_plot_perf(
            sheets,
            level="cells",
            title=tag.replace("_hoptimus_clusters_", " — ").replace("_5mean_5var", ""),
            figsize=(12, 6),
            savefig=os.path.join(plot_dir, f"{tag}_all_metrics.png"),
        )

    per_seed = pd.concat(tables, ignore_index=True)
    per_seed.to_csv(os.path.join(args.out_dir, "comparison_per_seed.csv"), index=False)

    metric = METRICS["balanced_accuracy"]
    summary = per_seed.groupby(["dataset", "method"], sort=False)[list(METRICS.values())].agg(["mean", ci95])
    summary.to_csv(os.path.join(args.out_dir, "comparison_summary.csv"))

    plot_balanced_accuracy(per_seed, os.path.join(plot_dir, "balanced_accuracy_30spots"))
    print(f"-> {os.path.join(plot_dir, 'balanced_accuracy_30spots.png')}")
    print(per_seed.groupby(["dataset", "method"], sort=False)[metric].agg(["mean", ci95, "size"]).to_string())


if __name__ == "__main__":
    main()
