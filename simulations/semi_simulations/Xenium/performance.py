"""
The performance figures of the Xenium semi-simulations: HEDeST against HEDeST + PPSA.

Three figures, both samples in each, from the ten seeds of ``hedest_code/xenium.yaml``:

* ``spot_pearson`` -- per cell type, the Pearson correlation over the spots between the mean
  prediction of a spot's cells and the spot's simulated proportions, which is
  :meth:`PredAnalyzer.spot_metrics`. Only the **held-out spots** (validation and test) of each
  seed are used, since the training spots are fitted by construction. PPSA reads each spot's
  own proportions, held-out ones included, so its bars are partly fitted by construction too.
* ``inside_outside_spots`` -- balanced accuracy over the cells inside spots and over the cells
  outside them, as the engine scored every run (``metrics.json``). PPSA is not gated: a cell
  outside the spots takes proportions interpolated from the nearest spots.
* ``pr_curves_{sample}`` -- per cell type, the one-vs-rest precision-recall curve of every
  scored cell against its ground truth type (argmax of ``sim_*_gt.csv``). Each seed's curve is
  read on a common recall grid as the interpolated precision, the best precision reached at that
  recall or beyond; the legend gives the average precision (``average_precision_score``, not
  interpolated) and the dashed line the type's prevalence, which is what a random ranking gets.

Statistics: bars and curves are the mean over the n = 10 seeds; error bars and shaded zones are
the t-based 95% CI of that mean, t(0.975, n - 1) x s / sqrt(n). No test is drawn.

Per-seed values and their summaries go to ``hedest_results/``; the figures, PNG and SVG, to
``figures/``.

Usage::

    python performance.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

import runs  # noqa: E402
from plots import _save  # noqa: E402
from plots import WIDTH_IN  # noqa: E402
from runs import SAMPLES  # noqa: E402
from runs import Sample  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "hedest_results"

#: ``(adjusted, label, colour)``: the two bars of Fig. 2 and of the MHAST benchmark.
VARIANTS: List[Tuple[bool, str, str]] = [(False, "HEDeST", "#9ecae1"), (True, "HEDeST + PPSA", "#08519c")]

#: The recall values every seed's PR curve is read at.
RECALL_GRID = np.linspace(0.0, 1.0, 201)

FONT = 7.0


def mean_ci(values: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean and half-width of its t-based 95% confidence interval, along ``axis``.

    Args:
        values: One row per seed (``axis=0``).
        axis: The seed axis.

    Returns:
        The mean and ``t(0.975, n - 1) * s / sqrt(n)``.
    """

    n = values.shape[axis]
    half = stats.t.ppf(0.975, n - 1) * values.std(axis=axis, ddof=1) / np.sqrt(n)

    return values.mean(axis=axis), half


def _style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FONT, length=2.5, width=0.6)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)


def _variant_handles() -> List[Patch]:
    return [Patch(facecolor=colour, label=label) for _, label, colour in VARIANTS]


# ----------------------------------------------------------------------------------------- tables


def spot_pearson_table() -> pd.DataFrame:
    """
    Per sample, seed, variant and cell type: Pearson r over the held-out spots.

    Returns:
        One row per (sample, seed, variant, cell type), with the number of spots.
    """

    from hedest.analysis.pred_analyzer import PredAnalyzer

    rows = []
    for key, sample in SAMPLES.items():
        for seed_dir in runs.seed_dirs(sample):
            for adjusted, label, _ in VARIANTS:
                analyzer = PredAnalyzer(runs.seed_run(sample, seed_dir, adjusted), adjusted=adjusted)
                table = analyzer.spot_metrics(subset="held_out").drop(index="__global__")
                for cell_type, values in table.iterrows():
                    rows.append(
                        {
                            "sample": key,
                            "seed": int(seed_dir.name[5:]),
                            "variant": label,
                            "cell_type": cell_type,
                            "pearson": values["pearson"],
                            "n_spots": table.attrs["n_spots"],
                        }
                    )
        logger.info(f"{sample.label}: spot-level Pearson of {len(runs.seed_dirs(sample))} seeds")

    return pd.DataFrame(rows)


def inside_outside_table() -> pd.DataFrame:
    """
    Per sample, seed and variant: balanced accuracy inside and outside spots, from ``metrics.json``.

    Returns:
        One row per (sample, seed, variant, where), with the number of cells.
    """

    import json

    rows = []
    for key, sample in SAMPLES.items():
        for seed_dir in runs.seed_dirs(sample):
            with open(seed_dir / "metrics.json") as handle:
                metrics = json.load(handle)
            for adjusted, label, _ in VARIANTS:
                for where in ("in_spot", "out_spot"):
                    rows.append(
                        {
                            "sample": key,
                            "seed": metrics["seed"],
                            "variant": label,
                            "where": where,
                            "balanced_accuracy": metrics[f"bal_acc_{'ppsa' if adjusted else 'raw'}_{where}"],
                            "n_cells": metrics[f"n_{where}"],
                        }
                    )

    return pd.DataFrame(rows)


def interpolated_precision(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    The precision-recall curve read on :data:`RECALL_GRID`.

    At every recall the precision is the best one reached at that recall or beyond, the usual
    interpolation, which makes the curve monotone and lets curves of different seeds be
    averaged point by point.

    Args:
        y_true: Whether each cell is of the type.
        scores: The probability of the type.

    Returns:
        The precision at each recall of the grid.
    """

    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, scores)
    # recall decreases along the arrays, so the best precision at recall >= r is a running max.
    best = np.maximum.accumulate(precision)[::-1]
    recall = recall[::-1]

    return best[np.searchsorted(recall, RECALL_GRID, side="left").clip(max=len(recall) - 1)]


def pr_tables() -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], np.ndarray]]:
    """
    Per sample, seed, variant and cell type: average precision and the interpolated curve.

    Returns:
        The average precisions (one row per (sample, seed, variant, cell type), with the
        prevalence), and the curves keyed by ``(sample, variant, cell type)``, seeds x grid.
    """

    from sklearn.metrics import average_precision_score

    rows, curves = [], {}
    for key, sample in SAMPLES.items():
        truth = runs.true_labels(sample)
        for seed_dir in runs.seed_dirs(sample):
            for adjusted, label, _ in VARIANTS:
                predictions = runs.seed_run(sample, seed_dir, adjusted).predictions
                labels = truth.reindex(predictions.index).to_numpy()
                for cell_type in predictions.columns:
                    y_true = labels == cell_type
                    scores = predictions[cell_type].to_numpy()
                    rows.append(
                        {
                            "sample": key,
                            "seed": int(seed_dir.name[5:]),
                            "variant": label,
                            "cell_type": cell_type,
                            "average_precision": float(average_precision_score(y_true, scores)),
                            "prevalence": float(y_true.mean()),
                        }
                    )
                    curves.setdefault((key, label, cell_type), []).append(interpolated_precision(y_true, scores))
        logger.info(f"{sample.label}: PR curves of {len(runs.seed_dirs(sample))} seeds")

    return pd.DataFrame(rows), {name: np.vstack(stack) for name, stack in curves.items()}


def summarise(table: pd.DataFrame, keys: List[str], value: str, carry: str) -> pd.DataFrame:
    """
    Mean and t-based 95% CI over the seeds of ``value``, per group.

    Args:
        table: One row per seed and group.
        keys: The columns naming a group.
        value: The column to summarise.
        carry: A column with one value per group, kept as it is (the mean if the seeds differ).

    Returns:
        One row per group.
    """

    rows = []
    for group, sub in table.groupby(keys, sort=False):
        mean, ci = mean_ci(sub[value].to_numpy())
        rows.append(
            {
                **dict(zip(keys, group)),
                carry: sub[carry].iloc[0] if sub[carry].nunique() == 1 else sub[carry].mean(),
                "n_seeds": len(sub),
                value: float(mean),
                f"{value}_ci": float(ci),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- figures


def _wrap(name: str) -> str:
    """A cell type name on two lines when it is long, so that seven of them fit side by side."""

    name = name.replace("_", " ")
    if "/" in name:
        return name.replace("/", "/\n", 1)
    if len(name) > 12 and " " in name:
        head, tail = name.rsplit(" ", 1)
        return f"{head}\n{tail}"

    return name


def _bar_pairs(ax: plt.Axes, summary: pd.DataFrame, groups: List[str], group_key: str, value: str) -> None:
    """Two bars per group, HEDeST then HEDeST + PPSA, with their 95% CI."""

    width = 0.38
    positions = np.arange(len(groups))
    for offset, (_, label, colour) in zip((-0.5, 0.5), VARIANTS):
        here = summary[summary.variant == label].set_index(group_key).loc[groups]
        ax.bar(
            positions + offset * width,
            here[value],
            width,
            yerr=here[f"{value}_ci"],
            color=colour,
            error_kw={"elinewidth": 0.7, "ecolor": "0.25", "capsize": 1.5, "capthick": 0.7},
        )
    ax.set_xticks(positions)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def figure_spot_pearson(summary: pd.DataFrame) -> plt.Figure:
    """Pearson r per cell type, one row of bars per sample."""

    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_IN, 4.4))
    for ax, (key, sample) in zip(axes, SAMPLES.items()):
        here = summary[summary["sample"] == key]
        cell_types = list(sample.colors)
        _bar_pairs(ax, here, cell_types, "cell_type", "pearson")
        ax.set_xticklabels([_wrap(name) for name in cell_types], fontsize=FONT)
        low = min(0.0, float((here.pearson - here.pearson_ci).min()) - 0.05)
        ax.set_ylim(low, 1.0)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.axhline(0, color="0.3", linewidth=0.6)
        n_spots = int(here.n_spots.iloc[0])
        ax.set_ylabel(f"{sample.label}\nPearson r", fontsize=FONT)
        ax.set_title(f"{sample.label}: {n_spots} held-out spots per seed", fontsize=FONT, loc="left")
        _style(ax)

    fig.suptitle(
        "Spot level: Pearson correlation between the mean prediction of a spot's cells and its proportions\n"
        "held-out spots; mean over 10 seeds, error bars = 95% CI",
        fontsize=FONT + 1,
        y=0.995,
    )
    fig.legend(
        handles=_variant_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        fontsize=FONT,
    )
    # Placed by hand: tight_layout already makes room for the suptitle, and a rect on top of it
    # left an empty band under the legend.
    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.08, right=0.99, hspace=0.6)

    return fig


def figure_inside_outside(summary: pd.DataFrame) -> plt.Figure:
    """Balanced accuracy inside and outside spots, one panel per sample."""

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN * 0.6, 2.6), sharey=True)
    for ax, (key, sample) in zip(axes, SAMPLES.items()):
        here = summary[summary["sample"] == key]
        _bar_pairs(ax, here, ["in_spot", "out_spot"], "where", "balanced_accuracy")
        counts = here.drop_duplicates("where").set_index("where").n_cells
        ax.set_xticklabels(
            [f"inside spots\n{counts['in_spot']:,} cells", f"outside spots\n{counts['out_spot']:,} cells"],
            fontsize=FONT,
        )
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.set_title(sample.label, fontsize=FONT + 1)
        _style(ax)
    axes[0].set_ylabel("balanced accuracy", fontsize=FONT)

    fig.suptitle(
        "Balanced accuracy of the cells inside and outside spots\nmean over 10 seeds, error bars = 95% CI",
        fontsize=FONT + 1,
        y=0.995,
    )
    fig.legend(
        handles=_variant_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=2, frameon=False, fontsize=FONT
    )
    fig.subplots_adjust(top=0.7, bottom=0.17, left=0.11, right=0.99, wspace=0.12)

    return fig


def figure_pr_curves(
    sample: Sample, key: str, ap: pd.DataFrame, curves: Dict[Tuple[str, str, str], np.ndarray]
) -> plt.Figure:
    """One panel per cell type: the two mean PR curves and their 95% CI zones."""

    cell_types = list(sample.colors)
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH_IN, 3.9), sharex=True, sharey=True)
    for ax, cell_type in zip(axes.flat, cell_types):
        lines = []
        for _, label, colour in VARIANTS:
            mean, ci = mean_ci(curves[(key, label, cell_type)])
            ax.fill_between(RECALL_GRID, mean - ci, np.minimum(mean + ci, 1.0), color=colour, alpha=0.3, lw=0)
            ax.plot(RECALL_GRID, mean, color=colour, lw=1.1)
            row = ap[(ap["sample"] == key) & (ap.variant == label) & (ap.cell_type == cell_type)].iloc[0]
            lines.append(f"AP {row.average_precision:.3f} ± {row.average_precision_ci:.3f}")
        prevalence = float(row.prevalence)
        ax.axhline(prevalence, color="0.5", ls="--", lw=0.7)

        for i, (text, (_, _, colour)) in enumerate(zip(lines, VARIANTS)):
            ax.text(
                0.03,
                0.05 + 0.1 * (1 - i),
                text,
                transform=ax.transAxes,
                fontsize=FONT - 1,
                color=colour,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
            )
        ax.set_title(f"{cell_type.replace('_', ' ')}\n{prevalence:.1%} of cells", fontsize=FONT)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        _style(ax)
        ax.grid(axis="x", color="0.9", linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("recall", fontsize=FONT)
    for ax in axes[:, 0]:
        ax.set_ylabel("precision", fontsize=FONT)

    legend = axes.flat[-1]
    legend.axis("off")
    handles = _variant_handles() + [Line2D([], [], color="0.5", ls="--", lw=0.7, label="prevalence (random ranking)")]
    legend.legend(handles=handles, loc="center", frameon=False, fontsize=FONT)

    fig.suptitle(
        f"{sample.label}: precision-recall per cell type, every scored cell\n"
        "mean over 10 seeds, zones = 95% CI; AP = average precision, mean ± 95% CI",
        fontsize=FONT + 1,
    )
    fig.tight_layout()

    return fig


# ----------------------------------------------------------------------------------------- driver


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    pearson = spot_pearson_table()
    pearson.to_csv(RESULTS / "spot_pearson.csv", index=False)
    pearson_summary = summarise(pearson, ["sample", "variant", "cell_type"], "pearson", carry="n_spots")
    pearson_summary.to_csv(RESULTS / "spot_pearson_summary.csv", index=False)
    _save(figure_spot_pearson(pearson_summary), "spot_pearson")

    inside = inside_outside_table()
    inside.to_csv(RESULTS / "inside_outside.csv", index=False)
    inside_summary = summarise(inside, ["sample", "variant", "where"], "balanced_accuracy", carry="n_cells")
    inside_summary.to_csv(RESULTS / "inside_outside_summary.csv", index=False)
    _save(figure_inside_outside(inside_summary), "inside_outside_spots")

    ap, curves = pr_tables()
    ap.to_csv(RESULTS / "average_precision.csv", index=False)
    ap_summary = summarise(ap, ["sample", "variant", "cell_type"], "average_precision", carry="prevalence")
    ap_summary.to_csv(RESULTS / "average_precision_summary.csv", index=False)
    for key, sample in SAMPLES.items():
        _save(figure_pr_curves(sample, key, ap_summary, curves), f"pr_curves_{key}")

    logger.info(f"Tables in {RESULTS}")


if __name__ == "__main__":
    main()
