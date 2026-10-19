"""
Figures of the Visium HD HEDeST study (see visiumhd.yaml): what the supervision signal costs.

The bars, the error bars and the confusion matrices are drawn by ``gridsearch/plots.py``'s own
functions, so these sit beside ``gridsearch/plots/`` and read the same way. What is new here is
the axis: the gridsearch compares *combinations*, this compares *supervisions* — the same
combination and the same ten seeds trained on the sample's own proportions, on the cell2location
proportions, and on proportions perturbed at four strengths.

Into ``{plots_dir}/``:

* ``supervision.png``        balanced accuracy with and without PPSA, gt against deconvolved, one
                             panel per sample; the number above each pair is what deconvolution
                             costs at that level
* ``ppsa_effect.png``, ``ppsa_gated.png``, ``ppsa_split.png``, ``ppsa_traintest.png``
                             the gridsearch's four PPSA figures, on the gt and deconvolved arms,
                             so the effect of PPSA can be read across levels and supervisions
* ``ppsa_gain.png``          the same effect as a single number per (level, supervision): does
                             PPSA help more when the proportions are wrong, and more when there
                             are more cell types?
* ``ppsa_effect_perturbation.png``  the PPSA figure again, on the perturbed arms
* ``perturbation_accuracy.png``     balanced accuracy against perturbation strength, with the
                             deconvolved arm drawn as a line across it: the accuracy reading of
                             the equivalence (``equivalence.py`` does the distance reading)
* ``per_class_recall.png``   per cell type, recall under gt supervision against recall under
                             deconvolved supervision — which types pay for it
* ``bias_vs_recall.png``     the mechanism: how much the deconvolution over- or under-called a
                             type against how much recall the model lost by training on it
* ``confusion/{dataset}_{level}.png``  recall matrices without / with PPSA and their difference,
                             summed over the ten seeds, for the gt and deconvolved arms

and ``{results_dir}/summary.csv``, one row per (sample, level, supervision).

Usage: python plots.py [--config visiumhd.yaml] [--svg]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = HERE.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "gridsearch"))

import plots as G  # noqa: E402  (gridsearch/plots.py)
from gridsearch import load_config  # type: ignore[attr-defined] # noqa: E402

sys.path.insert(0, str(STUDY / "deconv_code"))
import paths as D  # noqa: E402  (deconv_code/paths.py: the samples' own files)

SAMPLES = ["P1", "P2", "P5"]
LEVELS = ["lv0", "lv1", "lv2"]
STRENGTHS = [0.05, 0.1, 0.25, 0.5]
GT, DECONV = "#2171b5", "#d08b2f"
LIGHT_GT, LIGHT_DECONV = "#9ecae1", "#fdd0a2"


# ------------------------------------------------------------------------------------- the inputs


def load_runs(cfg: dict) -> pd.DataFrame:
    """``runs.csv``, collected first when it is missing or older than the runs."""

    runs_csv = Path(cfg["results_dir"]) / "runs.csv"
    newest = max(
        (p.stat().st_mtime for p in Path(cfg["out_dir"]).glob("*/*/*/*/seed_*/metrics.json")),
        default=0.0,
    )
    if not runs_csv.exists() or runs_csv.stat().st_mtime < newest:
        subprocess.run(
            [sys.executable, str(REPO / "gridsearch" / "gridsearch.py"), "collect", cfg["_path"]], check=True
        )
    runs = pd.read_csv(runs_csv)

    runs["sample"] = runs.dataset.str.split("_").str[0]
    runs["supervision"] = runs.group
    runs["strength"] = runs.dataset.str.extract(r"perturb([\d.]+)$")[0].astype(float).fillna(0.0)

    return runs


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI over seeds, per unit — the gridsearch's own aggregation."""

    summary = G.seed_means(runs, [])
    summary["sample"] = summary.dataset.str.split("_").str[0]
    summary["supervision"] = summary.group
    summary["strength"] = summary.dataset.str.extract(r"perturb([\d.]+)$")[0].astype(float).fillna(0.0)

    order = {level: i for i, level in enumerate(LEVELS)}

    return summary.sort_values(["sample", "strength", "level"], key=lambda s: s.map(order) if s.name == "level" else s)


def wide(summary: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    One row per (sample, level) with a column per (supervision, metric).

    ``unit_barplot`` wants one row per thing on the x axis and one column per bar. Here the
    thing on the x axis is a (sample, level) and the bars are the supervisions, which is the
    transpose of what the gridsearch plots — hence the pivot.
    """

    rows = []
    for (name, level), group in summary[summary.supervision.isin(["gt", "deconv"])].groupby(
        ["sample", "level"], sort=False
    ):
        row = {"group": name, "dataset": name, "level": level, "n_classes": int(group.n_classes.iloc[0])}
        for supervision in ["gt", "deconv"]:
            here = group[group.supervision == supervision]
            for column in columns:
                prefix = f"{supervision}_{column.replace('bal_acc_', '')}"
                row[prefix] = float(here[column].iloc[0]) if len(here) else np.nan
                row[f"{prefix}_ci"] = float(here[f"{column}_ci"].iloc[0]) if len(here) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- the figures


def figure_supervision(summary: pd.DataFrame, out: Path) -> None:
    """The headline: what training on deconvolved proportions costs, with and without PPSA."""

    table = wide(summary, ["bal_acc_raw", "bal_acc_ppsa"])
    G.unit_barplot(
        table,
        [
            ("gt_raw", "true proportions", LIGHT_GT),
            ("deconv_raw", "deconvolved proportions", LIGHT_DECONV),
            ("gt_ppsa", "true proportions + PPSA", GT),
            ("deconv_ppsa", "deconvolved proportions + PPSA", DECONV),
        ],
        "HEDeST trained on true vs deconvolved spot proportions\n"
        "same cells, same labels, same combination; error bars = 95% CI over 10 seeds; "
        "numbers = deconvolved − true",
        str(out / "supervision.png"),
        deltas=[("gt_raw", "deconv_raw"), ("gt_ppsa", "deconv_ppsa")],
    )


def figure_ppsa(summary: pd.DataFrame, out: Path) -> None:
    """The gridsearch's four PPSA figures, with supervision where it had combinations."""

    note = "error bars = 95% CI over 10 seeds"
    arms = summary[summary.supervision.isin(["gt", "deconv"])]
    perturbed = summary[summary.supervision == "perturb"]

    for table, stem, what in [
        (arms, "ppsa_effect", "true and deconvolved"),
        (perturbed, "ppsa_effect_perturbation", "perturbed"),
    ]:
        if table.empty:
            continue
        G.unit_barplot(
            table,
            [("bal_acc_raw", "without PPSA", G.LIGHT), ("bal_acc_ppsa", "with PPSA", G.DARK)],
            f"balanced accuracy without / with PPSA — {what} proportions\n{note}",
            str(out / f"{stem}.png"),
            deltas=[("bal_acc_raw", "bal_acc_ppsa")],
        )
    if arms.empty:
        return

    G.unit_barplot(
        arms,
        [
            ("bal_acc_raw", "no PPSA", G.LIGHT),
            ("bal_acc_ppsa", "PPSA everywhere", G.DARK),
            ("bal_acc_gated", "gated (inside spots only)", "#238b45"),
            ("bal_acc_antigated", "anti-gated (outside spots only)", "#cb181d"),
        ],
        f"where PPSA is applied\n{note}",
        str(out / "ppsa_gated.png"),
    )

    for stem, cols, labels, title in [
        (
            "ppsa_split",
            ["bal_acc_raw_in_spot", "bal_acc_ppsa_in_spot", "bal_acc_raw_out_spot", "bal_acc_ppsa_out_spot"],
            ["inside spots, no PPSA", "inside spots, PPSA", "outside spots, no PPSA", "outside spots, PPSA"],
            "cells inside vs outside spots",
        ),
        (
            "ppsa_traintest",
            ["bal_acc_raw_train_spot", "bal_acc_ppsa_train_spot", "bal_acc_raw_test_spot", "bal_acc_ppsa_test_spot"],
            ["train spots, no PPSA", "train spots, PPSA", "test spots, no PPSA", "test spots, PPSA"],
            "cells of the training spots vs of the test spots",
        ),
    ]:
        shades = ["#c6dbef", "#08519c", "#fdd0a2", "#d94801"]
        G.unit_barplot(
            arms,
            list(zip(cols, labels, shades)),
            f"{title}\n{note}",
            str(out / f"{stem}.png"),
            deltas=[(cols[0], cols[1]), (cols[2], cols[3])],
        )


def gain_table(runs: pd.DataFrame) -> pd.DataFrame:
    """
    PPSA's gain per unit, with the confidence interval of the *paired* difference.

    The two accuracies of a seed come from the same trained model, so they are paired: the
    interval that means anything is the one of their difference over the seeds, not the
    interval of either arm. Taking it from one arm would overstate the uncertainty of the gain
    by as much as the seeds disagree about the model itself.
    """

    runs = runs.copy()
    runs["gain"] = runs.bal_acc_ppsa - runs.bal_acc_raw
    grouped = runs.groupby(["sample", "level", "supervision", "strength", "n_classes"], sort=False)["gain"]
    table = pd.DataFrame(
        {"gain": grouped.mean(), "gain_ci": G.grouped_ci95(grouped), "n_seeds": grouped.size()}
    ).reset_index()

    return table


def figure_ppsa_gain(runs: pd.DataFrame, out: Path) -> None:
    """PPSA's gain as one number per (sample, level, supervision): does wrong supervision change it?"""

    arms = gain_table(runs)
    arms = arms[arms.supervision.isin(["gt", "deconv"])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    width = 0.35
    positions = np.arange(len(SAMPLES) * len(LEVELS))
    labels = [f"{s}\n{lv}" for s in SAMPLES for lv in LEVELS]

    for offset, (supervision, colour) in enumerate([("gt", GT), ("deconv", DECONV)]):
        values, errors = [], []
        for name in SAMPLES:
            for level in LEVELS:
                here = arms[(arms["sample"] == name) & (arms.level == level) & (arms.supervision == supervision)]
                values.append(float(here.gain.iloc[0]) if len(here) else np.nan)
                errors.append(float(here.gain_ci.iloc[0]) if len(here) else np.nan)
        axes[0].bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            color=colour,
            label={"gt": "true proportions", "deconv": "deconvolved"}[supervision],
            yerr=errors,
            capsize=2,
            error_kw=dict(lw=0.7, ecolor="#444"),
        )
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(positions, labels, fontsize=8)
    axes[0].set_ylabel("balanced accuracy with PPSA − without")
    axes[0].set_title("PPSA gain per sample and level (95% CI of the paired difference)", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Against the number of cell types, which is what a level really varies.
    for supervision, colour in [("gt", GT), ("deconv", DECONV)]:
        here = arms[arms.supervision == supervision]
        axes[1].scatter(here.n_classes, here.gain, color=colour, s=28, label=supervision)
        for _, row in here.iterrows():
            axes[1].annotate(
                f"{row['sample']} {row.level}",
                (row.n_classes, row.gain),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("number of cell types")
    axes[1].set_ylabel("PPSA gain")
    axes[1].set_title("PPSA gain against the size of the vocabulary", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Effect of PPSA across annotation levels and supervisions", fontsize=12)
    fig.tight_layout()
    G.save(fig, str(out / "ppsa_gain.png"))


def figure_perturbation(summary: pd.DataFrame, out: Path) -> None:
    """Balanced accuracy against perturbation strength, with the deconvolved arm across it."""

    fig, axes = plt.subplots(len(SAMPLES), len(LEVELS), figsize=(4.8 * len(LEVELS), 3.7 * len(SAMPLES)), squeeze=False)
    for row, name in enumerate(SAMPLES):
        for col, level in enumerate(LEVELS):
            ax = axes[row][col]
            here = summary[(summary["sample"] == name) & (summary.level == level)]
            gt = here[here.supervision == "gt"]
            deconv = here[here.supervision == "deconv"]
            perturbed = here[here.supervision == "perturb"].sort_values("strength")
            if gt.empty or perturbed.empty:
                ax.axis("off")
                continue

            for metric, colour, label in [
                ("bal_acc_raw", "0.35", "without PPSA"),
                ("bal_acc_ppsa", G.DARK, "with PPSA"),
            ]:
                x = np.concatenate([[0.0], perturbed.strength.to_numpy(float)])
                y = np.concatenate([gt[metric].to_numpy(float), perturbed[metric].to_numpy(float)])
                e = np.concatenate([gt[f"{metric}_ci"].to_numpy(float), perturbed[f"{metric}_ci"].to_numpy(float)])
                ax.errorbar(x, y, yerr=e, color=colour, marker="o", ms=4, lw=1.3, capsize=2, label=label)
            if not deconv.empty:
                ax.axhline(
                    float(deconv.bal_acc_raw.iloc[0]), color=DECONV, ls="--", lw=1.3, label="deconvolved (no PPSA)"
                )

            ax.set_title(f"{name} {level} ({int(gt.n_classes.iloc[0])} types)", fontsize=10)
            ax.set_xlabel("perturbation strength s", fontsize=8)
            ax.set_ylabel("balanced accuracy", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False)

    fig.suptitle(
        "Balanced accuracy against the perturbation of the spot proportions\n"
        "dashed: the accuracy reached with the cell2location proportions",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    G.save(fig, str(out / "perturbation_accuracy.png"))


def recall_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-type recall of every arm, from the confusion matrices summed over the seeds."""

    rows = []
    for (dataset, level), group in runs.groupby(["dataset", "level"], sort=False):
        classes: list[str] = []
        summed: dict[str, np.ndarray] = {}
        for path in group.path:
            with open(Path(path) / "metrics.json") as handle:
                confusion = json.load(handle)["confusion"]
            classes = confusion["classes"]
            for key in ["raw", "ppsa"]:
                matrix = np.asarray(confusion[key], dtype=float)
                summed[key] = summed[key] + matrix if key in summed else matrix

        for key, matrix in summed.items():
            support = matrix.sum(axis=1)
            recall = np.diag(matrix) / np.maximum(support, 1)
            for i, kind in enumerate(classes):
                rows.append(
                    {
                        "sample": dataset.split("_")[0],
                        "supervision": group.supervision.iloc[0],
                        "strength": float(group.strength.iloc[0]),
                        "level": level,
                        "variant": key,
                        "cell_type": kind,
                        "recall": float(recall[i]),
                        "support": float(support[i]),
                    }
                )

    return pd.DataFrame(rows)


def figure_per_class(recalls: pd.DataFrame, out: Path) -> None:
    """Per cell type, recall under true supervision against recall under deconvolved."""

    here = recalls[(recalls.variant == "raw") & (recalls.supervision.isin(["gt", "deconv"]))]
    pivot = here.pivot_table(index=["sample", "level", "cell_type"], columns="supervision", values="recall")
    if pivot.empty or not {"gt", "deconv"}.issubset(pivot.columns):
        print("    per_class_recall needs both arms, skipped")
        return
    pivot = pivot.reset_index()

    fig, axes = plt.subplots(len(SAMPLES), len(LEVELS), figsize=(4.4 * len(LEVELS), 3.6 * len(SAMPLES)), squeeze=False)
    for row, name in enumerate(SAMPLES):
        for col, level in enumerate(LEVELS):
            ax = axes[row][col]
            sub = pivot[(pivot["sample"] == name) & (pivot.level == level)]
            if sub.empty:
                ax.axis("off")
                continue
            drop = sub["gt"] - sub["deconv"]  # not sub.gt: that is DataFrame.gt
            order = np.argsort(drop.to_numpy())
            kinds = sub.cell_type.to_numpy()[order]
            ax.barh(
                range(len(sub)),
                drop.to_numpy()[order],
                color=[DECONV if d > 0 else GT for d in drop.to_numpy()[order]],
                edgecolor="0.3",
                linewidth=0.4,
            )
            ax.axvline(0, color="k", lw=0.8)
            ax.set_yticks(range(len(sub)), kinds, fontsize=7)
            ax.set_xlabel("recall lost (true − deconvolved)", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.set_title(f"{name} {level}", fontsize=10)

    fig.suptitle(
        "Per cell type, what deconvolved supervision costs in recall (without PPSA)\n"
        "bars to the right: the type is recalled less when the proportions are deconvolved",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    G.save(fig, str(out / "per_class_recall.png"))


def figure_bias_vs_recall(recalls: pd.DataFrame, out: Path) -> pd.DataFrame:
    """
    Does HEDeST inherit the deconvolution's bias? Per cell type, how much the deconvolution
    over- or under-called it against how much recall the model lost by being trained on it.
    """

    here = recalls[(recalls.variant == "raw") & (recalls.supervision.isin(["gt", "deconv"]))]
    wide_recall = here.pivot_table(index=["sample", "level", "cell_type"], columns="supervision", values="recall")
    if wide_recall.empty or not {"gt", "deconv"}.issubset(wide_recall.columns):
        return pd.DataFrame()
    wide_recall = wide_recall.reset_index()
    wide_recall["lost"] = wide_recall["gt"] - wide_recall["deconv"]

    rows = []
    for name in SAMPLES:
        sample = D.sample(name)
        for level in LEVELS:
            vocabulary = sample.vocabulary(level)
            true = pd.read_csv(sample.truth(level), index_col=0)[vocabulary]
            deconv = pd.read_csv(D.proportions_path(name, level), index_col=0)[vocabulary]
            for kind in vocabulary:
                rows.append(
                    {
                        "sample": name,
                        "level": level,
                        "cell_type": kind,
                        "true_share": float(true[kind].mean()),
                        "deconv_share": float(deconv[kind].mean()),
                    }
                )
    shares = pd.DataFrame(rows)
    # A ratio of means, in log2, so over- and under-calling are symmetric around 0. The epsilon
    # keeps a type the deconvolution never predicts from going to -inf.
    shares["log_ratio"] = np.log2((shares.deconv_share + 1e-6) / (shares.true_share + 1e-6))
    merged = wide_recall.merge(shares, on=["sample", "level", "cell_type"])

    rho = merged.log_ratio.corr(merged.lost, method="spearman")
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"lv0": "o", "lv1": "s", "lv2": "^"}
    for level, group in merged.groupby("level"):
        ax.scatter(
            group.log_ratio,
            group.lost,
            s=34,
            marker=markers.get(str(level), "o"),
            alpha=0.8,
            label=f"{level} (n={len(group)})",
        )
    for _, row in merged.reindex(merged.lost.abs().nlargest(8).index).iterrows():
        ax.annotate(
            f"{row['sample']} {row.cell_type}",
            (row.log_ratio, row.lost),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("log2(mean deconvolved share / mean true share)   — left: under-called, right: over-called")
    ax.set_ylabel("recall lost (true − deconvolved supervision)")
    ax.set_title(
        f"HEDeST inherits the deconvolution's per-type bias\nSpearman {rho:+.2f} over "
        f"{len(merged)} (sample, level, cell type) triples",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    G.save(fig, str(out / "bias_vs_recall.png"))

    return merged


def figures_confusion(runs: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    """The gridsearch's confusion figure for every gt and deconvolved arm."""

    (out / "confusion").mkdir(parents=True, exist_ok=True)
    for _, row in summary[summary.supervision.isin(["gt", "deconv"])].iterrows():
        here = runs[(runs.dataset == row.dataset) & (runs.level == row.level)]
        title = (
            f"{row.dataset} / {row.level} ({len(here)} seeds summed)\n"
            f"balanced accuracy {row.bal_acc_raw:.3f} without PPSA, {row.bal_acc_ppsa:.3f} with PPSA; "
            "rows = true type, normalised per row (diagonal = recall)"
        )
        G.confusion_plot(list(here.path), title, str(out / "confusion" / f"{row.dataset}_{row.level}.png"))


# ----------------------------------------------------------------------------------------- driver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "visiumhd.yaml"))
    parser.add_argument("--svg", action="store_true", help="also save every figure as SVG")
    arguments = parser.parse_args()

    G.SAVE_SVG = arguments.svg
    cfg = load_config(arguments.config)
    out = Path(cfg["plots_dir"])
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(cfg)
    summary = summarise(runs)
    print(f"  {len(runs)} runs over {summary.shape[0]} units")

    results = Path(cfg["results_dir"])
    keep = [
        "sample",
        "supervision",
        "strength",
        "level",
        "n_classes",
        "n_seeds",
        "bal_acc_raw",
        "bal_acc_raw_ci",
        "bal_acc_ppsa",
        "bal_acc_ppsa_ci",
    ]
    summary[keep].to_csv(results / "summary.csv", index=False)
    recalls = recall_table(runs)
    recalls.to_csv(results / "per_class_recall.csv", index=False)
    gain_table(runs).to_csv(results / "ppsa_gain.csv", index=False)

    figure_supervision(summary, out)
    figure_ppsa(summary, out)
    figure_ppsa_gain(runs, out)
    figure_perturbation(summary, out)
    figure_per_class(recalls, out)
    bias = figure_bias_vs_recall(recalls, out)
    if not bias.empty:
        bias.to_csv(results / "bias_vs_recall.csv", index=False)
    figures_confusion(runs, summary, out)

    print(f"  wrote {results / 'summary.csv'} and the figures in {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
