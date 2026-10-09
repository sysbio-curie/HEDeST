"""Figures of the segmentation comparison: the two segmentations against the three proportion methods.

Reads the tables of ``segmentation_collect.py`` and writes, in ``segmentation/plots/{config}/``:

    overview/accuracy_overview        every method averaged over the samples, samples behind the bar
    overview/accuracy_by_sample       the five methods as grouped bars, one group per sample
    overview/gain_over_segmentation   HEDeST minus the better of HoVerNet and CellViT
    per_class/recall_by_type          recall of each broad category, methods x types
    confusion/{sample}                row-normalised 3 x 3 confusion matrix of each method
    cell_set/filters                  what each filter of the shared cell set costs

Every figure is png + svg. The three proportion-based methods carry the 95% CI of their 3
seeds; HoVerNet and CellViT are deterministic and have none.

    python benchmark/comparison_code/segmentation_plots.py              # both configurations
    python benchmark/comparison_code/segmentation_plots.py --config gt
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from plots import _grid  # noqa: E402  (same gridlines as the four-method figures)
from plots import save_fig  # noqa: E402

TITLE = "STHELAR level 0, 3 broad types"


def _methods(frame: pd.DataFrame) -> list:
    """The methods present in a table, in ``config.SEG_METHODS`` order."""

    return [m for m in C.SEG_METHODS if m in set(frame["method"])]


def _subtitle(summary: pd.DataFrame, config: str) -> str:
    """One line naming the configuration, the samples and the cells behind a figure."""

    n_cells = f"{int(summary.drop_duplicates('sample')['n_cells'].sum()):,}".replace(",", " ")

    return (
        f"{TITLE} - {C.CONFIGS[config]}\n"
        f"{summary['sample'].nunique()} samples, {n_cells} cells scored by all "
        f"{len(_methods(summary))} methods"
    )


def plot_overview(summary: pd.DataFrame, stem: str, subtitle: str) -> None:
    """
    Every method averaged over the samples, with each sample drawn behind the bar.

    Args:
        summary: The per-method summary of one configuration.
        stem: Path without extension.
        subtitle: Figure title.
    """

    methods = _methods(summary)
    wide = summary.pivot_table(index="sample", columns="method", values="mean")[methods]
    means = wide.mean()

    fig, ax = plt.subplots(figsize=(1.7 * len(methods) + 3.0, 5.2))
    rng = np.random.default_rng(0)
    for i, method in enumerate(methods):
        ax.bar(i, means[method], 0.62, color=C.SEG_METHODS[method], zorder=2)
        jitter = (rng.random(len(wide)) - 0.5) * 0.28
        ax.scatter(i + jitter, wide[method], s=16, color="0.25", alpha=0.6, zorder=3, linewidths=0)
        ax.text(i, means[method] + 0.02, f"{means[method]:.3f}", ha="center", fontsize=9)

    ax.axhline(1 / len(C.BROAD_TYPES), color="#d94801", linestyle="--", linewidth=1.1, zorder=1)
    ax.text(len(methods) - 0.45, 1 / len(C.BROAD_TYPES) + 0.012, "chance", color="#d94801", fontsize=8, ha="right")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{subtitle}\nbar = mean over the samples, dots = one sample", fontsize=10)
    _grid(ax)
    save_fig(fig, stem)


def plot_by_sample(summary: pd.DataFrame, stem: str, subtitle: str) -> None:
    """
    The five methods as grouped bars, one group per sample and one for the mean.

    Args:
        summary: The per-method summary of one configuration.
        stem: Path without extension.
        subtitle: Figure title.
    """

    methods = _methods(summary)
    wide = summary.pivot_table(index="sample", columns="method", values="mean")[methods]
    errors = summary.pivot_table(index="sample", columns="method", values="ci")[methods]
    types = summary.drop_duplicates("sample").set_index("sample")["n_true_types"]

    samples = list(wide.index) + ["mean"]
    wide = pd.concat([wide, wide.mean().to_frame("mean").T])
    errors = pd.concat([errors, pd.DataFrame(0.0, index=["mean"], columns=methods)])

    x = np.arange(len(samples))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(1.55 * len(samples) + 3.0, 5.4))
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            wide[method],
            width,
            yerr=errors[method],
            color=C.SEG_METHODS[method],
            label=method,
            error_kw={"elinewidth": 1.1, "ecolor": "0.25", "capsize": 2.5},
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=2, rotation=90)

    ax.axvline(len(samples) - 1.5, color="0.75", linewidth=1, linestyle=":")
    labels = [f"{s}\n{int(types[s])} types" if s in types.index else "mean\n " for s in samples]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1.14)
    ax.set_title(subtitle, fontsize=10)
    _grid(ax)
    ax.legend(frameon=False, loc="upper center", ncol=len(methods), fontsize=8.5)
    save_fig(fig, stem)


def plot_gain(summary: pd.DataFrame, stem: str, subtitle: str) -> None:
    """
    HEDeST minus the better of HoVerNet and CellViT, per sample.

    Args:
        summary: The per-method summary of one configuration.
        stem: Path without extension.
        subtitle: Figure title.
    """

    wide = summary.pivot_table(index="sample", columns="method", values="mean")
    rivals = [m for m in C.SEG_SEGMENTERS if m in wide.columns]
    if "HEDeST" not in wide.columns or not rivals:
        return

    gain = (wide["HEDeST"] - wide[rivals].max(axis=1)).sort_values()
    colors = [C.SEG_METHODS["HEDeST"] if value > 0 else "#cb181d" for value in gain]

    fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(gain) + 2.5), 4.8))
    bars = ax.bar(range(len(gain)), gain.to_numpy(), 0.62, color=colors)
    ax.bar_label(bars, fmt="%+.3f", fontsize=8, padding=2)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_xticks(range(len(gain)))
    ax.set_xticklabels(gain.index, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("balanced accuracy difference")
    ax.set_title(
        f"{subtitle}\nHEDeST minus the better of HoVerNet and CellViT "
        f"({int((gain > 0).sum())}/{len(gain)} in favour of HEDeST)",
        fontsize=10,
    )
    ax.margins(y=0.18)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save_fig(fig, stem)


def plot_recall_by_type(confusion: pd.DataFrame, stem: str, subtitle: str) -> None:
    """
    Recall of each broad category, methods as bar groups, averaged over samples and seeds.

    Pooling the confusion counts over the samples weights a sample by its cell count; the
    recall is taken per sample and then averaged, so every sample counts once.

    Args:
        confusion: The long-form confusion table of one configuration.
        stem: Path without extension.
        subtitle: Figure title.
    """

    methods = _methods(confusion)
    per_sample = confusion.groupby(["sample", "method", "seed", "true"], sort=False)["n"].sum()
    hits = confusion[confusion["true"] == confusion["pred"]].set_index(["sample", "method", "seed", "true"])["n"]
    recall = (hits / per_sample).replace([np.inf, -np.inf], np.nan).dropna()
    recall = recall[per_sample.reindex(recall.index) > 0]
    mean = recall.groupby(["method", "true"]).mean().unstack("true")

    types = [t for t in C.BROAD_TYPES if t in mean.columns]
    x = np.arange(len(types))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(2.3 * len(types) + 3.0, 5.0))
    for i, method in enumerate(methods):
        if method not in mean.index:
            continue
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, mean.loc[method, types], width, color=C.SEG_METHODS[method], label=method)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=10)
    ax.set_ylabel("recall, mean over samples and seeds")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{subtitle}\nper-category recall", fontsize=10)
    _grid(ax)
    ax.legend(frameon=False, loc="upper center", ncol=len(methods), fontsize=8.5)
    save_fig(fig, stem)


def plot_confusions(confusion: pd.DataFrame, out_dir: str, subtitle: str) -> int:
    """
    One figure per sample: the row-normalised confusion matrix of each method.

    Args:
        confusion: The long-form confusion table of one configuration.
        out_dir: Folder for the per-sample figures.
        subtitle: Figure title, the sample is prepended.

    Returns:
        How many figures were written.
    """

    methods = _methods(confusion)
    written = 0

    for sample in sorted(confusion["sample"].unique()):
        sub = confusion[confusion["sample"] == sample]
        fig, axes = plt.subplots(1, len(methods), figsize=(2.9 * len(methods), 3.6), squeeze=False)

        for index, method in enumerate(methods):
            ax = axes[0][index]
            counts = (
                sub[sub["method"] == method]
                .groupby(["true", "pred"], sort=False)["n"]
                .sum()
                .unstack("pred")
                .reindex(index=C.BROAD_TYPES, columns=C.BROAD_TYPES)
                .fillna(0.0)
            )
            present = counts.sum(axis=1) > 0  # lymph_node_s0 has no Epithelial truth
            rates = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)

            ax.imshow(rates.to_numpy(), cmap="Blues", vmin=0, vmax=1)
            for i in range(len(C.BROAD_TYPES)):
                for j in range(len(C.BROAD_TYPES)):
                    value = rates.to_numpy()[i, j]
                    if np.isnan(value):
                        ax.text(j, i, "-", ha="center", va="center", fontsize=9, color="0.6")
                        continue
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="white" if value > 0.55 else "0.15",
                    )
            ax.set_xticks(range(len(C.BROAD_TYPES)))
            ax.set_xticklabels([t[:5] for t in C.BROAD_TYPES], fontsize=8)
            ax.set_yticks(range(len(C.BROAD_TYPES)))
            # Only the leftmost panel carries the category names; repeating them would run
            # them into the panel on the left.
            if index == 0:
                labels = [t if ok else f"{t}\n(absent)" for t, ok in zip(C.BROAD_TYPES, present)]
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_ylabel("true", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.set_title(method, fontsize=10, color="0.15")
            ax.set_xlabel("predicted", fontsize=8)

        first = sub[sub["method"] == methods[0]]  # one method, one seed = the shared cell set once
        n_cells = f"{int(first[first['seed'] == first['seed'].iloc[0]]['n'].sum()):,}".replace(",", " ")
        fig.suptitle(f"{sample} - {subtitle.splitlines()[0]}\n{n_cells} cells, row-normalised", fontsize=10)
        fig.subplots_adjust(wspace=0.35)
        save_fig(fig, os.path.join(out_dir, sample))
        written += 1

    return written


def plot_cell_set(cell_set: pd.DataFrame, stem: str, subtitle: str) -> None:
    """
    What each filter of the shared cell set costs, per sample.

    Args:
        cell_set: The cell-set table of one configuration.
        stem: Path without extension.
        subtitle: Figure title.
    """

    steps = [
        ("n_hovernet", "all HoVerNet nuclei", "#d9d9d9"),
        ("n_matched", "matched to a CellViT nucleus", "#bcbddc"),
        ("n_matched_broad", "both classes are broad types", "#9e9ac8"),
        ("n_scored", "scored (ground truth + all methods)", C.SEG_METHODS["HEDeST"]),
    ]
    table = cell_set.set_index("sample").sort_index()
    x = np.arange(len(table))
    width = 0.8 / len(steps)

    fig, ax = plt.subplots(figsize=(1.55 * len(table) + 3.0, 5.0))
    for i, (column, label, color) in enumerate(steps):
        offset = (i - (len(steps) - 1) / 2) * width
        bars = ax.bar(x + offset, table[column] / 1000.0, width, color=color, label=label)
        shares = table[column] / table["n_hovernet"]
        ax.bar_label(bars, labels=[f"{s:.0%}" for s in shares], fontsize=6.5, padding=2, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("cells (thousands)")
    ax.set_title(f"{subtitle}\nhow the shared cell set is built, % of all HoVerNet nuclei", fontsize=10)
    ax.margins(y=0.16)
    _grid(ax, step=100)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    save_fig(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=list(C.CONFIGS) + ["both"], default="both", help="Which configuration.")
    parser.add_argument("--result-dir", default=C.SEG_DIR, help="Where segmentation_collect.py wrote its tables.")
    parser.add_argument("--plot-dir", default=C.SEG_PLOT_DIR, help="Where the figures go.")
    args = parser.parse_args()

    for config in list(C.CONFIGS) if args.config == "both" else [args.config]:
        path = os.path.join(args.result_dir, f"summary_{config}.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} is missing; run segmentation_collect.py --config {config} first")
            continue

        summary = pd.read_csv(path)
        confusion = pd.read_csv(os.path.join(args.result_dir, f"confusion_{config}.csv"))
        cell_set = pd.read_csv(os.path.join(args.result_dir, f"cell_set_{config}.csv"))
        subtitle = _subtitle(summary, config)
        out = os.path.join(args.plot_dir, config)

        plot_overview(summary, os.path.join(out, "overview", "accuracy_overview"), subtitle)
        plot_by_sample(summary, os.path.join(out, "overview", "accuracy_by_sample"), subtitle)
        plot_gain(summary, os.path.join(out, "overview", "gain_over_segmentation"), subtitle)
        plot_recall_by_type(confusion, os.path.join(out, "per_class", "recall_by_type"), subtitle)
        plot_cell_set(cell_set, os.path.join(out, "cell_set", "filters"), subtitle)
        n_confusion = plot_confusions(confusion, os.path.join(out, "confusion"), subtitle)
        print(f"{config}: 5 figures + {n_confusion} confusion figures -> {out}/..")


if __name__ == "__main__":
    main()
