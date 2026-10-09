"""Figures comparing the four methods on the STHELAR samples, for one proportion configuration.

Reads the tables of ``collect.py`` and writes, in ``plots/{config}/``:

    accuracy_by_level     one panel per sample, balanced accuracy against the annotation level
    accuracy_overview     every method averaged over the sample-levels, with the spread
    ranking               mean rank of each method across the sample-levels
    gain_over_best_rival  HEDeST + PPSA minus the better of HistoCell and PanoSpace

Every figure is png + svg, with the 3 seeds as a 95% confidence interval.

    python benchmark/comparison_code/plots.py                # both configurations
    python benchmark/comparison_code/plots.py --config gt
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import config as C  # noqa: E402


def save_fig(fig: plt.Figure, stem: str, dpi: int = 200) -> None:
    """Saves ``fig`` as ``stem.png`` and ``stem.svg`` and closes it."""

    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _grid(ax: plt.Axes, step: float = 0.1) -> None:
    """Horizontal gridline every ``step``, under the data."""

    ax.yaxis.set_major_locator(plt.MultipleLocator(step))
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _legend(fig: plt.Figure, methods: Sequence[str], extra: Sequence[tuple] = ()) -> None:
    """One shared legend under the figure."""

    entries = [(m, C.METHODS[m]) for m in methods] + list(extra)
    handles = [Line2D([], [], marker="s", linestyle="", markersize=9, color=c) for _, c in entries]
    fig.legend(
        handles,
        [label for label, _ in entries],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(entries),
    )


def plot_by_level(summary: pd.DataFrame, stem: str, title: str) -> None:
    """
    Balanced accuracy against the annotation level, one panel per sample.

    Args:
        summary: The per-method summary of one configuration.
        stem: Path without extension.
        title: Figure title.
    """

    samples = sorted(summary["sample"].unique())
    methods = [m for m in C.METHODS if m in set(summary["method"])]
    cols = min(4, len(samples))
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.4 * rows), squeeze=False, sharey=True)

    for index, sample in enumerate(samples):
        ax = axes.ravel()[index]
        last_row = index >= len(samples) - cols  # only the bottom panels carry the x label
        sub = summary[summary["sample"] == sample]
        for method in methods:
            part = sub[sub["method"] == method].sort_values("level_num")
            if part.empty:
                continue
            ax.errorbar(
                part["level_num"],
                part["mean"],
                yerr=part["ci"],
                marker="o",
                markersize=4.5,
                capsize=3,
                linewidth=1.6,
                color=C.METHODS[method],
            )
        types = sub.drop_duplicates("level_num").sort_values("level_num")
        ax.plot(types["level_num"], 1 / types["n_types"], color="#d94801", linestyle="--", linewidth=1.1)
        ax.set_xticks(sorted(sub["level_num"].unique()))
        ax.set_title(f"{sample}  ({', '.join(str(n) for n in types['n_types'])} types)", fontsize=9)
        if last_row:
            ax.set_xlabel("annotation level", fontsize=8)
        ax.set_ylim(0, 1)
        _grid(ax)

    for ax in axes.ravel()[len(samples) :]:
        ax.set_visible(False)
    for r in range(rows):
        axes[r][0].set_ylabel("balanced accuracy")

    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.33)
    _legend(fig, methods, extra=[("chance (1/types)", "#d94801")])
    save_fig(fig, stem)


def plot_overview(summary: pd.DataFrame, stem: str, title: str) -> None:
    """
    Every method averaged over the sample-levels, with each sample-level drawn behind the bar.

    Args:
        summary: The per-method summary of one configuration.
        stem: Path without extension.
        title: Figure title.
    """

    methods = [m for m in C.METHODS if m in set(summary["method"])]
    wide = summary.pivot_table(index=["sample", "level"], columns="method", values="mean")[methods]
    fig, ax = plt.subplots(figsize=(1.7 * len(methods) + 3.5, 5.2))

    means = wide.mean()
    rng = np.random.default_rng(0)
    for i, method in enumerate(methods):
        ax.bar(i, means[method], 0.62, color=C.METHODS[method], zorder=2)
        jitter = (rng.random(len(wide)) - 0.5) * 0.28
        ax.scatter(i + jitter, wide[method], s=11, color="0.25", alpha=0.55, zorder=3, linewidths=0)
        ax.text(i, means[method] + 0.02, f"{means[method]:.3f}", ha="center", fontsize=9)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{title}\nbar = mean over the {len(wide)} sample-levels, dots = one sample-level", fontsize=10)
    _grid(ax)
    save_fig(fig, stem)


def plot_ranking(summary: pd.DataFrame, stem: str, title: str) -> None:
    """
    Mean rank of each method across the sample-levels (1 = best), with its 95% CI.

    Ranking within each sample-level puts every dataset on the same footing whatever its
    number of cell types, as in the gridsearch study.

    Args:
        summary: The per-method summary of one configuration.
        stem: Path without extension.
        title: Figure title.
    """

    methods = [m for m in C.METHODS if m in set(summary["method"])]
    wide = summary.pivot_table(index=["sample", "level"], columns="method", values="mean")[methods]
    ranks = wide.rank(axis=1, ascending=False)
    means, n = ranks.mean(), len(ranks)
    from scipy import stats

    ci = stats.t.ppf(0.975, n - 1) * ranks.std(ddof=1) / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(1.7 * len(methods) + 3.5, 4.6))
    order = means.sort_values().index
    ax.bar(
        range(len(order)),
        [means[m] for m in order],
        0.62,
        yerr=[ci[m] for m in order],
        color=[C.METHODS[m] for m in order],
        error_kw={"elinewidth": 1.2, "ecolor": "0.25", "capsize": 4},
    )
    for i, method in enumerate(order):
        wins = int((ranks[method] == 1).sum())
        ax.text(i, 0.08, f"{means[method]:.2f}\n{wins}/{n} wins", ha="center", fontsize=8, color="white")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("mean rank (1 = best)")
    ax.set_ylim(0, len(methods) + 0.3)
    ax.invert_yaxis()
    ax.set_title(f"{title}\nranked within each of the {n} sample-levels", fontsize=10)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save_fig(fig, stem)


def plot_per_sample(summary: pd.DataFrame, out_dir: str, title: str) -> int:
    """
    One figure per sample: the four methods as grouped bars, one group per annotation level.

    The by-level curves put every sample in one figure; this is the same data read the other
    way round, one sample at a time, where the per-level gaps between methods are easier to
    read off.

    Args:
        summary: The per-method summary of one configuration.
        out_dir: Folder for the per-sample figures.
        title: Figure title, the sample is prepended.

    Returns:
        How many figures were written.
    """

    methods = [m for m in C.METHODS if m in set(summary["method"])]
    written = 0

    for sample in sorted(summary["sample"].unique()):
        sub = summary[summary["sample"] == sample]
        levels = sorted(sub["level_num"].unique())
        x = np.arange(len(levels))
        width = 0.8 / len(methods)

        fig, ax = plt.subplots(figsize=(1.5 * len(levels) + 4.0, 5.0))
        for i, method in enumerate(methods):
            part = sub[sub["method"] == method].set_index("level_num").reindex(levels)
            offset = (i - (len(methods) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                part["mean"],
                width,
                yerr=part["ci"],
                color=C.METHODS[method],
                label=method,
                error_kw={"elinewidth": 1.1, "ecolor": "0.25", "capsize": 2.5},
            )
            ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=2)

        types = sub.drop_duplicates("level_num").set_index("level_num").reindex(levels)
        ax.plot(x, 1 / types["n_types"], color="#d94801", linestyle="--", linewidth=1.2, label="chance (1/types)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"level {lvl}\n{int(n)} types" for lvl, n in zip(levels, types["n_types"])], fontsize=9)
        ax.set_ylabel("balanced accuracy")
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{sample} - {title}", fontsize=11)
        _grid(ax)
        ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8.5)
        save_fig(fig, os.path.join(out_dir, sample))
        written += 1

    return written


def plot_gain(summary: pd.DataFrame, stem: str, title: str) -> None:
    """
    HEDeST + PPSA minus the better of HistoCell and PanoSpace, per sample-level.

    Args:
        summary: The per-method summary of one configuration.
        stem: Path without extension.
        title: Figure title.
    """

    wide = summary.pivot_table(index=["sample", "level"], columns="method", values="mean")
    rivals = [m for m in ("HistoCell", "PanoSpace") if m in wide.columns]
    if "HEDeST + PPSA" not in wide.columns or not rivals:
        return

    gain = (wide["HEDeST + PPSA"] - wide[rivals].max(axis=1)).sort_values()
    labels = [f"{s} {lvl.replace('level', 'L')}" for s, lvl in gain.index]
    colors = ["#08519c" if v > 0 else "#cb181d" for v in gain]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.32 * len(gain) + 3), 4.8))
    ax.bar(range(len(gain)), gain.values, 0.72, color=colors)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_xticks(range(len(gain)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("balanced accuracy difference")
    ax.set_title(
        f"{title}\nHEDeST + PPSA minus the better of HistoCell and PanoSpace "
        f"({int((gain > 0).sum())}/{len(gain)} in favour of HEDeST)",
        fontsize=10,
    )
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save_fig(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=list(C.CONFIGS) + ["both"], default="both", help="Which configuration.")
    parser.add_argument("--result-dir", default=C.RESULT_DIR, help="Where collect.py wrote its tables.")
    parser.add_argument("--plot-dir", default=C.PLOT_DIR, help="Where the figures go.")
    args = parser.parse_args()

    for config in list(C.CONFIGS) if args.config == "both" else [args.config]:
        path = os.path.join(args.result_dir, f"summary_{config}.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} is missing; run collect.py --config {config} first")
            continue

        summary = pd.read_csv(path)
        out = os.path.join(args.plot_dir, config, "overview")
        label = f"STHELAR, {C.CONFIGS[config]} (3 seeds, 95% CI)"
        n_units = summary[["sample", "level"]].drop_duplicates().shape[0]

        plot_by_level(summary, os.path.join(out, "accuracy_by_level"), label)
        plot_overview(summary, os.path.join(out, "accuracy_overview"), label)
        plot_ranking(summary, os.path.join(out, "ranking"), label)
        plot_gain(summary, os.path.join(out, "gain_over_best_rival"), label)
        per_sample = plot_per_sample(summary, os.path.join(args.plot_dir, config, "per_sample"), label)
        print(f"{config}: 4 overview + {per_sample} per-sample figures over {n_units} sample-levels -> {out}/..")


if __name__ == "__main__":
    main()
