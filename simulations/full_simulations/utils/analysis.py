"""Helpers to score and plot the HEDeST runs of the fully simulated datasets.

Everything that needs the predictions goes through the HEDeST analysis package:
:class:`hedest.analysis.PredAnalyzer` for the metrics, the confusion matrices and the
galleries, :func:`hedest.analysis.load_run` / :func:`hedest.analysis.load_seed_runs` for the
runs, and :class:`hedest.analysis.Palette` for the colours. The ground truth of these
datasets is a one-hot table, which ``PredAnalyzer.set_ground_truth`` reads directly.

Two levels of result are used, as in the gridsearch study:

* **per seed** — one run, one row per (seed, variant), for the curves and their CI;
* **aggregate** — the mean probabilities over the seeds (``load_run`` on the sample folder),
  for the confusion matrices and the galleries, so that one figure shows one model.

"variant" is ``raw`` (model output) or ``ppsa`` (after the adjustment). The colours of the
clusters are those of the construction figures (``cluster_colors``), so a cluster keeps its
colour from ``01_cluster.py`` to the confusion matrices.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from .data_simulation import cluster_colors
from .data_simulation import save_fig
from hedest.analysis import load_run
from hedest.analysis import load_seed_runs
from hedest.analysis import Palette
from hedest.analysis import PredAnalyzer

VARIANTS = {"raw": "no PPSA", "ppsa": "PPSA"}
VARIANT_COLORS = {"raw": "#9ecae1", "ppsa": "#08519c"}
STRENGTH_LABEL = "perturbation strength"
BA = "balanced_accuracy"


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------


def load_gt(sim_dir: str, tag: str) -> pd.DataFrame:
    """
    The one-hot ground truth of a dataset, indexed by cell id as a string.

    Args:
        sim_dir: The dataset folder.
        tag: The dataset tag.

    Returns:
        The ``cells x clusters`` table.
    """

    gt = pd.read_csv(os.path.join(sim_dir, f"{tag}_gt.csv"), index_col=0)
    gt.index = gt.index.astype(str)

    return gt


def load_images(sim_dir: str, tag: str) -> Dict[str, torch.Tensor]:
    """
    The 64 px / 20 um crops of the cells of a dataset, keyed by cell id as a string.

    Args:
        sim_dir: The dataset folder.
        tag: The dataset tag.

    Returns:
        ``{cell id: crop}``.
    """

    path = os.path.join(sim_dir, f"{tag}_image_dict_64px_20um.pt")

    return {str(cell): image for cell, image in torch.load(path).items()}


def palette_of(clusters: Sequence[str]) -> Palette:
    """
    The palette of a dataset: the cluster colours of the construction figures.

    Args:
        clusters: The cluster names, in the order of the proportion columns.

    Returns:
        The palette.
    """

    colors = cluster_colors(len(clusters))

    return Palette.from_colors({name: colors[i][:3] for i, name in enumerate(clusters)}, names=list(clusters))


def analyzer(
    run_dir: str,
    gt: pd.DataFrame,
    variant: str,
    image_dict: Optional[Dict[str, torch.Tensor]] = None,
    aggregate: bool = True,
) -> PredAnalyzer:
    """
    A :class:`PredAnalyzer` on one variant of a run or of a folder of seeds.

    Args:
        run_dir: A ``seed_*`` folder, or the sample folder holding them.
        gt: The one-hot ground truth.
        variant: ``"raw"`` or ``"ppsa"``.
        image_dict: The cell crops, for the galleries.
        aggregate: Whether to average the seeds of ``run_dir`` (ignored for a single run).

    Returns:
        The analyzer, with the ground truth and the cluster palette attached.
    """

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}, got '{variant}'")

    adjusted = variant == "ppsa"
    run = load_run(run_dir, adjusted=adjusted, aggregate=aggregate)

    # PredAnalyzer defaults to adjusted=True and would switch a raw run back, so it is explicit.
    return PredAnalyzer(run, image_dict=image_dict, ground_truth=gt, palette=palette_of(run.ct_list), adjusted=adjusted)


def seed_metrics(sample_dir: str, gt: pd.DataFrame) -> pd.DataFrame:
    """
    Cell-level metrics of every seed of a sample, for both variants.

    Args:
        sample_dir: The folder holding the ``seed_*`` runs.
        gt: The one-hot ground truth.

    Returns:
        One row per (seed, variant), with the metrics of ``PredAnalyzer.cell_metrics``.
    """

    rows = []
    for variant in VARIANTS:
        adjusted = variant == "ppsa"
        for run in load_seed_runs(sample_dir, adjusted=adjusted):
            metrics = PredAnalyzer(run, ground_truth=gt, adjusted=adjusted).cell_metrics(per_class=False)
            rows.append({"variant": variant, "seed": run.seeds[0], **metrics})

    if not rows:
        raise FileNotFoundError(f"No seed run under {sample_dir}")

    return pd.DataFrame(rows)


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


def summarise(results: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """
    Mean and 95% CI of the balanced accuracy over the seeds.

    Args:
        results: The per-seed table.
        keys: The columns identifying a group.

    Returns:
        One row per group, with ``mean``, ``ci`` and ``n_seeds``.
    """

    grouped = results.groupby(list(keys), sort=False, dropna=False)[BA]
    out = pd.DataFrame({"mean": grouped.mean(), "ci": grouped.apply(ci95), "n_seeds": grouped.size()}).reset_index()

    return out


def paired_t_test(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    """
    Two-sided paired t-test of ``a`` against ``b``, matched by seed, with a Wilcoxon check.

    Args:
        a: The scores of one method, indexed by seed.
        b: The scores of the other method, indexed by seed.

    Returns:
        ``n`` (paired seeds), ``mean_a``, ``mean_b``, ``diff`` (mean of ``a - b``), ``t``,
        ``df`` (``n - 1``), ``p``, and ``p_wilcoxon`` (two-sided Wilcoxon signed-rank test).
        NaN where a test is undefined, e.g. when ``a - b`` is the same on every seed.
    """

    from scipy import stats

    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = (pair["a"] - pair["b"]).to_numpy(dtype=float)
    n = len(d)
    t, p = stats.ttest_rel(pair["a"], pair["b"]) if n > 1 and np.ptp(d) > 0 else (np.nan, np.nan)
    try:
        p_wilcoxon = float(stats.wilcoxon(d).pvalue)
    except ValueError:  # every difference is zero
        p_wilcoxon = np.nan

    return {
        "n": n,
        "mean_a": pair["a"].mean(),
        "mean_b": pair["b"].mean(),
        "diff": d.mean(),
        "t": float(t),
        "df": n - 1,
        "p": float(p),
        "p_wilcoxon": p_wilcoxon,
    }


def ppsa_tests(results: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """
    PPSA against no PPSA in every group: paired t-test of the balanced accuracy over the seeds.

    Args:
        results: The per-seed table.
        keys: The columns identifying a group (not ``variant`` nor ``seed``).

    Returns:
        One row per group, with the fields of :func:`paired_t_test` (``a`` = PPSA,
        ``b`` = no PPSA, so a positive ``diff`` is a gain of PPSA).
    """

    rows = []
    for group, sub in results.groupby(list(keys), sort=False, dropna=False):
        wide = sub.pivot_table(index="seed", columns="variant", values=BA)
        if "raw" not in wide or "ppsa" not in wide:
            continue
        group = group if isinstance(group, tuple) else (group,)
        rows.append({**dict(zip(keys, group)), **paired_t_test(wide["ppsa"], wide["raw"])})

    return pd.DataFrame(rows)


def format_p(p: float) -> str:
    """
    A P value for a plot, two significant digits: decimals down to 0.001, then ``m × 10^e``.

    Args:
        p: The P value.

    Returns:
        The label, e.g. ``"P = 0.034"``, ``"P = 0.0014"`` or ``"P = 5.4 × 10$^{-6}$"`` (mathtext).
    """

    if not np.isfinite(p):
        return "P = n/a"
    if p >= 1e-3:
        return f"P = {p:.{max(2, 1 - int(np.floor(np.log10(p))))}f}"
    if p == 0:
        return "P < 10$^{-300}$"
    exponent = int(np.floor(np.log10(p)))
    mantissa = p / 10**exponent
    if round(mantissa, 1) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1

    return f"P = {mantissa:.1f} × 10$^{{{exponent}}}$"


# ----------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------


def _grid(ax: plt.Axes, step: float = 0.1) -> None:
    """Horizontal gridline every ``step``, under the data, as in the other studies."""

    ax.yaxis.set_major_locator(plt.MultipleLocator(step))
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def short_name(sample: str) -> str:
    """
    A sample name without the parts every sample shares, for an axis tick.

    Args:
        sample: The sample name.

    Returns:
        The shortened name.
    """

    return sample.replace("_hoptimus_clusters_", " clusters, ").replace("_15mean_15var", "").replace("_5mean_5var", "")


def _variant_legend(
    ax: plt.Axes,
    extra: Sequence[Tuple[str, str]] = (),
    figure: Optional[plt.Figure] = None,
    anchor: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Legend of the two variants, plus optional ``(label, colour)`` entries.

    Args:
        ax: The axes to draw it in, when it fits (curve plots leave the bottom left free).
        extra: Additional ``(label, colour)`` entries.
        figure: Given for a bar plot with short x labels: the legend goes under the whole
            figure instead of on top of the bars.
        anchor: Given for a bar plot with tall x labels: the legend goes above the axes, at
            these axes coordinates, between the title and the bars.
    """

    entries = [(VARIANTS[v], VARIANT_COLORS[v]) for v in VARIANTS] + list(extra)
    handles = [Line2D([], [], marker="s", linestyle="", markersize=9, color=c) for _, c in entries]
    labels = [label for label, _ in entries]

    if anchor is not None:
        ax.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=anchor, ncol=len(entries))
    elif figure is not None:
        figure.legend(
            handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=len(entries)
        )
    else:
        ax.legend(handles, labels, frameon=False, loc="lower left", ncol=2)


def plot_ppsa_bars(summary: pd.DataFrame, stem: str, title: str, chance: Optional[pd.Series] = None) -> None:
    """
    Balanced accuracy of every sample, without and with PPSA, as paired bars.

    Args:
        summary: Output of :func:`summarise` keyed by ``("sample", "variant")``.
        stem: Path without extension.
        title: Figure title.
        chance: Optional chance level per sample, drawn as a tick on each pair.
    """

    samples = list(dict.fromkeys(summary["sample"]))
    x = np.arange(len(samples))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(samples) + 3), 5.5))

    for i, variant in enumerate(VARIANTS):
        sub = summary[summary["variant"] == variant].set_index("sample").reindex(samples)
        ax.bar(
            x + (i - 0.5) * width,
            sub["mean"],
            width,
            yerr=sub["ci"],
            color=VARIANT_COLORS[variant],
            error_kw={"elinewidth": 1, "ecolor": "0.3"},
            label=VARIANTS[variant],
        )

    if chance is not None:
        for i, sample in enumerate(samples):
            value = chance.get(sample)
            if value is not None:
                ax.plot([x[i] - width, x[i] + width], [value, value], color="#d94801", linewidth=1.2, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(s) for s in samples], rotation=90, fontsize=7)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=34)
    _grid(ax)
    # The labels are tall and the bars reach the bottom, so the legend goes under the title.
    _variant_legend(
        ax,
        extra=[("chance (1/K)", "#d94801")] if chance is not None else [],
        anchor=(0.5, 1.015),
    )
    save_fig(fig, stem)


def plot_perturbation(summary: pd.DataFrame, stem: str, title: str) -> None:
    """
    Balanced accuracy against the perturbation strength, one panel per dataset.

    Args:
        summary: Output of :func:`summarise` keyed by ``("tag", "strength", "variant")``.
        stem: Path without extension.
        title: Figure title.
    """

    tags = list(dict.fromkeys(summary["tag"]))
    fig, axes = plt.subplots(1, len(tags), figsize=(3.6 * len(tags), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, tag in zip(axes, tags):
        for variant in VARIANTS:
            sub = summary[(summary["tag"] == tag) & (summary["variant"] == variant)].sort_values("strength")
            ax.errorbar(
                sub["strength"],
                sub["mean"],
                yerr=sub["ci"],
                marker="o",
                markersize=5,
                capsize=3,
                linewidth=1.6,
                color=VARIANT_COLORS[variant],
                label=VARIANTS[variant],
            )
        ax.set_title(tag.replace("_hoptimus_clusters_200spots", "").replace("_15mean_15var", ""), fontsize=9)
        ax.set_xlabel(STRENGTH_LABEL)
        ax.set_ylim(0, 1)
        _grid(ax)

    axes[0].set_ylabel("balanced accuracy")
    _variant_legend(axes[0])
    fig.suptitle(title)
    save_fig(fig, stem)


def plot_k_series(summary: pd.DataFrame, stem: str, title: str) -> None:
    """
    Balanced accuracy against the number of clusters, one panel per balance, chance level shown.

    Args:
        summary: Output of :func:`summarise` keyed by ``("K", "balance", "variant")``.
        stem: Path without extension.
        title: Figure title.
    """

    balances = list(dict.fromkeys(summary["balance"]))
    fig, axes = plt.subplots(1, len(balances), figsize=(4.6 * len(balances), 4.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, balance in zip(axes, balances):
        for variant in VARIANTS:
            sub = summary[(summary["balance"] == balance) & (summary["variant"] == variant)].sort_values("K")
            ax.errorbar(
                sub["K"],
                sub["mean"],
                yerr=sub["ci"],
                marker="o",
                markersize=5,
                capsize=3,
                linewidth=1.6,
                color=VARIANT_COLORS[variant],
                label=VARIANTS[variant],
            )
        ks = sorted(summary["K"].unique())
        ax.plot(ks, [1 / k for k in ks], color="#d94801", linestyle="--", linewidth=1.2, label="chance (1/K)")
        ax.set_xticks(ks)
        ax.set_xlabel("number of clusters K")
        ax.set_title(balance, fontsize=10)
        ax.set_ylim(0, 1)
        _grid(ax)

    axes[0].set_ylabel("balanced accuracy")
    _variant_legend(axes[0], extra=[("chance (1/K)", "#d94801")])
    fig.suptitle(title)
    save_fig(fig, stem)


def plot_design_bars(
    summary: pd.DataFrame,
    stem: str,
    title: str,
    order: Sequence[str],
    tests: Optional[pd.DataFrame] = None,
) -> None:
    """
    Balanced accuracy of a family of designs, grouped by balance.

    Args:
        summary: Output of :func:`summarise` keyed by ``("design", "balance", "variant")``.
        stem: Path without extension.
        title: Figure title.
        order: The designs, left to right.
        tests: Output of :func:`ppsa_tests` keyed by ``("design", "balance")``: every pair
            of bars gets a bracket with the P value of PPSA against no PPSA.
    """

    balances = list(dict.fromkeys(summary["balance"]))
    x = np.arange(len(order))
    width = 0.38
    fig, axes = plt.subplots(1, len(balances), figsize=(4.4 * len(balances), 4.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, balance in zip(axes, balances):
        for i, variant in enumerate(VARIANTS):
            sub = (
                summary[(summary["balance"] == balance) & (summary["variant"] == variant)]
                .set_index("design")
                .reindex(order)
            )
            ax.bar(
                x + (i - 0.5) * width,
                sub["mean"],
                width,
                yerr=sub["ci"],
                color=VARIANT_COLORS[variant],
                error_kw={"elinewidth": 1, "ecolor": "0.3"},
            )
        if tests is not None:
            tops = (
                summary[summary["balance"] == balance]
                .assign(top=lambda t: t["mean"] + t["ci"])
                .groupby("design")["top"]
                .max()
            )
            for j, design in enumerate(order):
                row = tests[(tests["balance"] == balance) & (tests["design"] == design)]
                if row.empty or design not in tops:
                    continue
                y = tops[design] + 0.03
                x1, x2 = j - 0.5 * width, j + 0.5 * width
                ax.plot([x1, x1, x2, x2], [y, y + 0.015, y + 0.015, y], color="black", lw=1)
                ax.text(j, y + 0.022, format_p(row["p"].iloc[0]), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=9)
        ax.set_title(balance, fontsize=10)
        ax.set_ylim(0, 1 if tests is None else 1.15)
        _grid(ax)
        ax.set_yticks(np.round(np.arange(0, 1.01, 0.1), 1))

    axes[0].set_ylabel("balanced accuracy")
    _variant_legend(axes[0], figure=fig)
    fig.suptitle(title)
    save_fig(fig, stem)


def plot_confusion(
    matrices: Dict[str, pd.DataFrame],
    stem: str,
    title: str,
    scores: Optional[Dict[str, float]] = None,
) -> None:
    """
    Row-normalised confusion matrices without PPSA, with PPSA, and their difference.

    Args:
        matrices: ``{"raw": matrix, "ppsa": matrix}``, counts, rows = ground truth.
        stem: Path without extension.
        title: Figure title.
        scores: Optional ``{variant: balanced accuracy}`` to print above each panel.
    """

    recall = {name: matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0) for name, matrix in matrices.items()}
    delta = recall["ppsa"] - recall["raw"]
    panels = [("raw", recall["raw"], "magma_r", 0, 1), ("ppsa", recall["ppsa"], "magma_r", 0, 1)]
    bound = float(np.nanmax(np.abs(delta.to_numpy()))) or 1.0
    panels.append(("ppsa - raw", delta, "coolwarm", -bound, bound))

    n = len(recall["raw"])
    size = max(3.2, 0.32 * n + 1.6)
    fig, axes = plt.subplots(1, 3, figsize=(3 * size + 1.5, size + 0.9))

    for ax, (name, matrix, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        label = VARIANTS.get(name, name)
        if scores is not None and name in scores:
            label = f"{label} — balanced accuracy {scores[name]:.3f}"
        ax.set_title(label, fontsize=9)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([c.replace("Cluster ", "") for c in matrix.columns], fontsize=7)
        ax.set_yticklabels([c.replace("Cluster ", "") for c in matrix.index], fontsize=7)
        ax.set_xlabel("predicted cluster", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("true cluster", fontsize=8)
        if n <= 10:
            for i in range(n):
                for j in range(n):
                    value = matrix.to_numpy()[i, j]
                    if np.isfinite(value) and abs(value) >= 0.01:
                        ax.text(
                            j,
                            i,
                            f"{value:.2f}".lstrip("0"),
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="0.2" if abs(value) < 0.6 else "white",
                        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=10)
    save_fig(fig, stem)


def plot_galleries(
    an: PredAnalyzer,
    gt: pd.DataFrame,
    stem: str,
    title: str,
    num_rows: int = 6,
    num_cols: int = 6,
) -> None:
    """
    Two galleries of the same cells: grouped by predicted cluster, then by true cluster.

    Only the ``png`` is written: the crops are bitmaps, so an ``svg`` would be the same
    pixels in a much larger file.

    Args:
        an: The analyzer, with its crops attached.
        gt: The one-hot ground truth.
        stem: Path without extension; ``_pred`` and ``_gt`` are appended.
        title: Figure title.
        num_rows: Rows of crops per cluster.
        num_cols: Columns of crops per cluster.
    """

    from hedest.analysis import plots

    os.makedirs(os.path.dirname(stem), exist_ok=True)
    capacity = num_rows * num_cols

    fig = an.plot_celltype_grid(num_rows=num_rows, num_cols=num_cols, selection="max")
    fig.suptitle(f"{title} — predicted clusters (most confident cells)", fontsize=10, y=1.02)
    fig.savefig(f"{stem}_pred.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    truth = gt.idxmax(axis=1)
    blocks: Dict[str, List[str]] = {}
    for name in an.ct_list:
        cells = [c for c in truth.index[truth == name] if c in an.image_dict]
        if cells:
            blocks[name] = list(pd.Series(cells).sample(n=min(capacity, len(cells)), random_state=0))

    fig = plots.plot_gallery(
        an.image_dict,
        blocks,
        num_rows=num_rows,
        num_cols=num_cols,
        palette=an.palette,
        counts={name: int((truth == name).sum()) for name in blocks},
    )
    fig.suptitle(f"{title} — true clusters (random cells)", fontsize=10, y=1.02)
    fig.savefig(f"{stem}_gt.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
