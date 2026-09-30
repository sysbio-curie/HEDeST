"""Figures of a HEDeST gridsearch (see gridsearch.py).

    python gridsearch/plots.py CONFIG [--svg]

Reads ``{results_dir}/runs.csv`` (``gridsearch.py collect`` is run first when it is
missing or older than the runs) and writes, for each feature type, in
``{plots_dir}/{feature}/``, the results of every combination (barplots, per_dataset),
their rankings, which select one combination, and the analysis of that combination on
every dataset (confusion, ppsa_*):

* ``barplots/{scope}_{ppsa|raw}.png``  balanced accuracy of every combination, mean over
  the datasets of the scope (+- 95% CI across datasets). Scopes: ``all``, each dataset
  group (``group_*``) and each range of number of cell types (``types_*``), so that
  datasets of comparable difficulty are compared together.
* ``per_dataset/{dataset}_{level}_{ppsa|raw}.png``  the same for one dataset
  (+- 95% CI across seeds).
* ``rankings/{scope}_{ppsa|raw}_{balacc|meanrank|compare}.png``  the combinations of each
  scope ranked two ways: by mean balanced accuracy, and by mean rank (ranked within every
  dataset-level, then averaged, so that a dataset with 3 cell types weighs as much as one
  with 12). The mean-rank figure carries the tests that elect the best combination
  (Friedman, Nemenyi critical difference, Wilcoxon signed-rank against the elected
  combination with Holm correction, see ``rank_stats``); ``compare`` puts the two
  rankings side by side for every combination.
* ``confusion/{dataset}_{level}.png``  recall matrices without PPSA, with PPSA and their
  difference, of the selected combination (summed over seeds).
* ``ppsa_effect.png``, ``ppsa_gated.png``, ``ppsa_split.png``, ``ppsa_traintest.png``
  per dataset, the selected combination without / with PPSA; PPSA everywhere vs only
  inside spots (gated) vs only outside spots (anti-gated); cells inside vs outside
  spots; cells of the training spots vs of the test spots.

and ``{plots_dir}/features.png`` comparing the feature types (each with its selected
combination) on the datasets they share.

The selected combination is the best mean rank over all datasets without PPSA
(``plots: {select: ppsa}`` elects it with PPSA, ``plots: {combination: {feature: name}}``
imposes it). Every bar of a dataset in the confusion and ppsa_* figures comes from the
same runs, the seeds of that combination.

In ``{results_dir}``: ``ranking_{feature}_{scope}_{variant}.csv`` (one row per
combination, both rankings and the test against the elected combination, sorted by mean
rank), ``ranking_tests.csv`` (one row per ranking: Friedman test, critical difference,
elected combination, agreement of the two rankings), ``selected.csv`` (the selected
combination of each feature type) and ``selected_{feature}.csv`` (its scores on every
dataset).
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import subprocess
import sys
from typing import Dict
from typing import List
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gridsearch import load_config  # noqa: E402

VARIANTS = {"ppsa": ("bal_acc_ppsa", "with PPSA"), "raw": ("bal_acc_raw", "without PPSA")}
GRID = dict(ls=":", lw=0.5, color="#bbb")
LIGHT, DARK = "#bdbdbd", "#2171b5"

SAVE_SVG = False


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def acc_axis(ax, axis: str = "y", lo: float = 0.0, hi: float = 1.0) -> None:
    """Balanced accuracy axis from 0 to 1, a gridline every 0.1."""

    (ax.yaxis if axis == "y" else ax.xaxis).set_major_locator(MultipleLocator(0.1))
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)
    ax.grid(axis=axis, **GRID)
    ax.set_axisbelow(True)


def save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(os.path.splitext(path)[0] + ".svg", bbox_inches="tight")
    plt.close(fig)


def fmt_value(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def sort_values(values) -> list:
    """Numbers numerically, hidden_dims ("512-256") by their layer sizes, the rest as text."""

    values = list(pd.unique(values))
    for key in (float, lambda v: [int(x) for x in str(v).split("-")], str):
        try:
            return sorted(values, key=key)
        except (TypeError, ValueError):
            continue
    return values


def fmt_p(p: float) -> str:
    return "p < 1e-300" if p == 0 else f"p = {p:.2g}"


def unit_label(row) -> str:
    return f"{row['dataset']}\n{row['level']} ({row['n_classes']})"


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def grouped_ci95(g) -> pd.DataFrame:
    """Half-width of the t-based 95% confidence interval of the mean of every group (0 if n < 2)."""

    n = g.count()
    half = stats.t.ppf(0.975, (n - 1).clip(lower=1)) * g.std(ddof=1) / np.sqrt(n)
    return half.where(n >= 2, 0.0)


def seed_means(runs: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Mean and 95% CI over seeds of every balanced accuracy, per (unit, combination)."""

    metrics = [c for c in runs.columns if c.startswith("bal_acc")]
    by = ["feature", "group", "dataset", "level", "n_classes", "combination", *keys]
    g = runs.groupby(by, sort=False, dropna=False)[metrics]
    out = pd.concat([g.mean(), grouped_ci95(g).add_suffix("_ci"), g.size().rename("n_seeds")], axis=1)
    return out.reset_index()


def scope_means(sm: pd.DataFrame, keys: List[str], metric: str) -> pd.DataFrame:
    """Per combination, mean and 95% CI across the units of the scope."""

    g = sm.groupby(["combination", *keys], sort=False, dropna=False)[metric]
    out = pd.DataFrame({"mean": g.mean(), "ci": grouped_ci95(g), "n_units": g.size()}).reset_index()
    return out.sort_values("mean", ascending=False)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""

    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    out = np.empty_like(adj)
    out[order] = adj
    return out


def rank_stats(sub: pd.DataFrame, keys: List[str], metric: str, alpha: float = 0.05):
    """
    Ranking by mean rank: the combinations are ranked within every dataset-level (1 = best,
    ties averaged) and sorted by their mean rank across the dataset-levels, so that every
    dataset-level weighs the same whatever its number of cell types.

    Tests (Demsar 2006; Benavoli et al. 2016):
      * Friedman test (and its Iman-Davenport F form): do the combinations differ at all?
      * Nemenyi critical difference: two mean ranks closer than CD are not distinguishable.
      * the combination with the best mean rank is elected; every other combination is
        compared with it by a one-sided Wilcoxon signed-rank test on the balanced
        accuracies, paired by dataset-level, Holm-corrected over all the comparisons.
        The combinations it is not significantly better than are "tied with the best".

    Returns:
        (table, tests): one row per combination (mean balanced accuracy and mean rank,
        position by each, test against the elected combination), sorted by mean rank;
        and a dict with the omnibus test, the elected combination and the comparison
        with the ranking by mean balanced accuracy.
    """

    wide = sub.pivot_table(index=["dataset", "level"], columns="combination", values=metric, aggfunc="first")
    wide = wide.dropna(axis=1, how="any")  # a combination missing on a dataset-level cannot be ranked there
    n, k = wide.shape
    ranks = wide.rank(axis=1, ascending=False)

    params = sub.drop_duplicates("combination").set_index("combination")[keys]
    t = pd.DataFrame(
        {
            "mean": wide.mean(),
            "ci": grouped_ci95(wide),
            "mean_rank": ranks.mean(),
            "mean_rank_ci": grouped_ci95(ranks),
        }
    )
    t = params.join(t, how="inner")
    t["pos_bal_acc"] = t["mean"].rank(ascending=False, method="min").astype(int)
    t = t.sort_values(["mean_rank", "mean"], ascending=[True, False])
    t["pos_mean_rank"] = np.arange(1, k + 1)
    best = t.index[0]

    diff = wide[best].values[:, None] - wide.values  # best - other, per dataset-level
    p = np.ones(k)
    for j, c in enumerate(wide.columns):
        if c != best and np.any(diff[:, j] != 0):
            p[j] = stats.wilcoxon(diff[:, j], alternative="greater").pvalue
    others = wide.columns != best
    p_holm = np.ones(k)
    p_holm[others] = holm(p[others])
    col = {c: j for j, c in enumerate(wide.columns)}
    idx = [col[c] for c in t.index]
    t["wins_vs_best"] = (diff < 0).sum(axis=0)[idx]  # dataset-levels where the combination beats the best
    t["losses_vs_best"] = (diff > 0).sum(axis=0)[idx]
    t["p_wilcoxon"] = p[idx]
    t["p_holm"] = p_holm[idx]
    t["tied_with_best"] = t["p_holm"] >= alpha
    t["n_units"] = n
    t = t.reset_index().rename(columns={"index": "combination"})

    chi2, p_friedman = stats.friedmanchisquare(*wide.values.T) if k >= 3 and n >= 2 else (np.nan, np.nan)
    f_id = (n - 1) * chi2 / (n * (k - 1) - chi2)
    p_id = stats.f.sf(f_id, k - 1, (k - 1) * (n - 1))
    cd = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    by_acc = t.sort_values("pos_bal_acc").iloc[0]
    top = min(10, k)
    tests = {
        "n_units": n,
        "n_combinations": k,
        "friedman_chi2": chi2,
        "friedman_p": p_friedman,
        "iman_davenport_F": f_id,
        "iman_davenport_p": p_id,
        "nemenyi_cd": cd,
        "best": best,
        "best_mean_rank": t["mean_rank"].iloc[0],
        "best_bal_acc": t["mean"].iloc[0],
        "best_pos_bal_acc": int(t["pos_bal_acc"].iloc[0]),
        "n_tied_with_best": int(t["tied_with_best"].sum()) - 1,  # the elected combination excluded
        # with n dataset-levels the smallest one-sided Wilcoxon p is 2^-n: below alpha / (k - 1),
        # Holm can never reject and every combination is tied with the best
        "can_reject": bool(0.5**n < alpha / max(k - 1, 1)),
        "best_by_bal_acc": by_acc["combination"],
        "best_by_bal_acc_pos_mean_rank": int(by_acc["pos_mean_rank"]),
        "best_by_bal_acc_tied": bool(by_acc["tied_with_best"]),
        "spearman": stats.spearmanr(t["mean"], -t["mean_rank"])[0],
        "top10_overlap": len(
            set(t.nsmallest(top, "pos_mean_rank")["combination"]) & set(t.nsmallest(top, "pos_bal_acc")["combination"])
        ),
    }
    return t, tests


def select_combination(sm: pd.DataFrame, keys: List[str], feature: str, cfg: dict) -> dict:
    """
    The combination analysed on every dataset (confusion and ppsa_* figures): the best mean
    rank over all the datasets of the feature type, without PPSA unless ``plots: {select:
    ppsa}``, or the one given in ``plots: {combination: {feature: name}}``.
    """

    opts = cfg.get("plots", {})
    variant = opts.get("select", "raw")
    if variant not in VARIANTS:
        raise ValueError(f"plots: select must be one of {list(VARIANTS)}, not {variant}")
    t, tests = rank_stats(sm, keys, VARIANTS[variant][0])
    combination = (opts.get("combination") or {}).get(feature)
    if combination is None:
        combination, how = tests["best"], f"best mean rank over all datasets {VARIANTS[variant][1]}"
    elif combination in set(sm["combination"]):
        how = "set in the configuration"
    else:
        raise ValueError(f"plots: combination {combination} of {feature} is not in the runs")
    r = t.set_index("combination").reindex([combination]).iloc[0]
    scores = sm.loc[sm["combination"] == combination, ["bal_acc_raw", "bal_acc_ppsa"]].mean()
    return {
        "feature": feature,
        "combination": combination,
        "chosen_by": how,
        "mean_rank": r["mean_rank"],
        "pos_mean_rank": r["pos_mean_rank"],
        "pos_bal_acc": r["pos_bal_acc"],
        "n_tied_with_best": tests["n_tied_with_best"],
        "n_units": int((sm["combination"] == combination).sum()),
        "bal_acc_raw": scores["bal_acc_raw"],
        "bal_acc_ppsa": scores["bal_acc_ppsa"],
    }


def scopes(sm: pd.DataFrame, class_bins: List[List[int]], min_units: int = 3) -> Dict[str, pd.DataFrame]:
    """
    The groups of units the combinations are averaged over. A scope needs `min_units`
    dataset-levels: with 2, the CI across them (t(1) = 12.7) says nothing.
    """

    out = {"all": sm}
    for group in pd.unique(sm["group"]):
        out[f"group_{str(group).replace(' ', '')}"] = sm[sm["group"] == group]
    for lo, hi in class_bins:
        out[f"types_{lo}-{hi}"] = sm[(sm["n_classes"] >= lo) & (sm["n_classes"] <= hi)]
    return {k: v for k, v in out.items() if v.groupby(["dataset", "level"]).ngroups >= min_units}


def scope_title(name: str, sub: pd.DataFrame) -> str:
    n = sub.groupby(["dataset", "level"]).ngroups
    if name == "all":
        what = "all datasets"
    elif name.startswith("group_"):
        what = f"{sub['group'].iloc[0]} datasets"
    else:
        what = f"datasets with {name[6:]} cell types"
    return f"{what} ({n} dataset-levels)"


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def facet_layout(keys: List[str]):
    """Rows, columns, x axis and bar colours of the combination barplots."""

    keys = list(keys)
    x = "lr" if "lr" in keys else keys[-1]
    rest = [k for k in keys if k != x]
    hue = "alpha" if "alpha" in rest else (rest[-1] if rest else None)
    rest = [k for k in rest if k != hue]
    cols = [k for k in ("divergence", "hidden_dims") if k in rest]
    rows = [k for k in rest if k not in cols]
    return rows, cols, x, hue


def combo_barplot(agg: pd.DataFrame, keys: List[str], title: str, path: str, ylabel: str) -> None:
    """Every combination: rows x columns of panels, x = one parameter, bars = another."""

    rows, cols, x, hue = facet_layout(keys)
    values = {k: sort_values(agg[k]) for k in keys}
    row_vals = list(itertools.product(*(values[k] for k in rows))) or [()]
    col_vals = list(itertools.product(*(values[k] for k in cols))) or [()]
    hue_vals = values[hue] if hue else [None]
    colors = plt.cm.Blues(np.linspace(0.25, 0.95, len(hue_vals)))
    width = 0.8 / len(hue_vals)

    fig, axes = plt.subplots(
        len(row_vals),
        len(col_vals),
        figsize=(1.2 + 5.2 * len(col_vals), 0.8 + 2.6 * len(row_vals)),
        sharey=True,
        squeeze=False,
    )
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            ax = axes[i, j]
            sub = agg
            for k, v in zip(rows + cols, rv + cv):
                sub = sub[sub[k] == v]
            for h, (hv, color) in enumerate(zip(hue_vals, colors)):
                s = sub if hue is None else sub[sub[hue] == hv]
                s = s.set_index(x).reindex(values[x])
                pos = np.arange(len(values[x])) + (h - (len(hue_vals) - 1) / 2) * width
                ax.bar(
                    pos,
                    s["mean"],
                    width,
                    yerr=s["ci"],
                    color=color,
                    capsize=2,
                    error_kw=dict(lw=0.7, ecolor="#444"),
                    label=f"{hue}={fmt_value(hv)}" if hue else None,
                )
            ax.set_xticks(np.arange(len(values[x])))
            ax.set_xticklabels([fmt_value(v) for v in values[x]], fontsize=8)
            acc_axis(ax)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0 and cols:
                ax.set_title(", ".join(f"{k} = {fmt_value(v)}" for k, v in zip(cols, cv)), fontsize=10)
            if j == 0:
                ax.set_ylabel(
                    (", ".join(f"{k}={fmt_value(v)}" for k, v in zip(rows, rv)) + "\n" if rows else "") + ylabel,
                    fontsize=9,
                )
            if i == len(row_vals) - 1:
                ax.set_xlabel(x)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, 1.0 - 0.25 / (0.8 + 2.6 * len(row_vals))),
        )
    fig.suptitle(title, y=1.0 + 0.1 / len(row_vals), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (0.8 + 2.6 * len(row_vals))))
    save(fig, path)


def ranking_plot(agg: pd.DataFrame, title: str, path: str, xlabel: str, top: int = 25) -> None:
    sub = agg.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.8 + 0.32 * len(sub)))
    ax.barh(
        np.arange(len(sub)), sub["mean"], xerr=sub["ci"], color=DARK, capsize=2, error_kw=dict(lw=0.7, ecolor="#444")
    )
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels([c.replace("_", " ") for c in sub["combination"]], fontsize=8)
    for k, m in enumerate(sub["mean"]):
        ax.text(m - 0.01, k, f"{m:.3f}", ha="right", va="center", fontsize=7, color="white")
    acc_axis(ax, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    save(fig, path)


def meanrank_plot(t: pd.DataFrame, tests: dict, title: str, path: str, top: int = 25) -> None:
    """The best mean ranks (+- 95% CI), coloured by the test against the elected combination."""

    sub = t.head(top).iloc[::-1]
    y = np.arange(len(sub))
    best = tests["best_mean_rank"]
    fig, ax = plt.subplots(figsize=(11, 1.6 + 0.32 * len(sub)))
    ax.axvline(
        best + tests["nemenyi_cd"],
        color="#d94801",
        ls="--",
        lw=1,
        label=f"elected + Nemenyi critical difference ({tests['nemenyi_cd']:.1f}): all pairs closer are not distinguishable",
    )
    for tied, color, label in [
        (True, DARK, "not significantly worse than the elected combination"),
        (False, LIGHT, "significantly worse (Wilcoxon signed-rank, Holm, p < 0.05)"),
    ]:
        s = sub["tied_with_best"].values == tied
        if s.any():
            ax.errorbar(
                sub["mean_rank"].values[s],
                y[s],
                xerr=sub["mean_rank_ci"].values[s],
                fmt="o",
                color=color,
                ms=5,
                capsize=2,
                elinewidth=0.8,
                label=label,
            )
    ax.plot(best, y[-1], marker="*", ms=14, color="#d94801", ls="none", label="elected (best mean rank)")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in sub["combination"]], fontsize=8)
    for yy, (_, r) in zip(y, sub.iterrows()):
        ax.annotate(
            f"bal. acc. {r['mean']:.3f} (#{r['pos_bal_acc']})   wins/losses vs elected "
            f"{r['wins_vs_best']}/{r['losses_vs_best']}   p_Holm {r['p_holm']:.2g}",
            (1.01, yy),
            xycoords=("axes fraction", "data"),
            va="center",
            fontsize=7,
            color="#444",
        )
    lo = max(0.0, (sub["mean_rank"] - sub["mean_rank_ci"]).min() - 3)
    ax.set_xlim(lo, max((sub["mean_rank"] + sub["mean_rank_ci"]).max(), best + tests["nemenyi_cd"]) * 1.05)
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(
        f"mean rank across dataset-levels (+- 95% CI; 1 = best of {tests['n_combinations']}, lower is better)"
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.9 / (1.6 + 0.32 * len(sub)) - 0.02),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    note = (
        ""
        if tests["can_reject"]
        else (
            f"\ntoo few dataset-levels ({tests['n_units']}) for Holm over "
            f"{tests['n_combinations'] - 1} comparisons to reject anything"
        )
    )
    ax.set_title(
        f"{title}\nFriedman chi2 = {tests['friedman_chi2']:.0f}, {fmt_p(tests['friedman_p'])}; "
        f"{tests['n_tied_with_best']} other combinations not significantly worse than the elected one{note}",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, path)


def compare_plot(t: pd.DataFrame, tests: dict, title: str, path: str) -> None:
    """Every combination: mean balanced accuracy vs mean rank."""

    fig, ax = plt.subplots(figsize=(8, 6.5))
    tied = t["tied_with_best"].values
    ax.scatter(
        t["mean"][~tied], t["mean_rank"][~tied], s=10, color=LIGHT, label="significantly worse than the elected one"
    )
    ax.scatter(t["mean"][tied], t["mean_rank"][tied], s=12, color=DARK, label="not significantly worse")
    marks = [
        (t.iloc[0], "*", "#d94801", 200, "best mean rank (elected)"),
        (t.sort_values("pos_bal_acc").iloc[0], "D", "#6a3d9a", 50, "best mean balanced accuracy"),
    ]
    for r, marker, color, size, label in marks:
        ax.scatter(
            r["mean"],
            r["mean_rank"],
            marker=marker,
            s=size,
            color=color,
            edgecolor="white",
            lw=0.5,
            zorder=3,
            label=f"{label}: {r['combination'].replace('_', ' ')}",
        )
    ax.invert_yaxis()
    acc_axis(ax, axis="x", lo=np.floor(t["mean"].min() * 10) / 10, hi=np.ceil(t["mean"].max() * 10) / 10)
    ax.grid(axis="y", **GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("mean balanced accuracy across dataset-levels")
    ax.set_ylabel("mean rank across dataset-levels (1 = best)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8)
    ax.set_title(
        f"{title}\nSpearman rho = {tests['spearman']:.3f}, top 10 in common: {tests['top10_overlap']}/10", fontsize=10
    )
    fig.tight_layout()
    save(fig, path)


def unit_barplot(
    table: pd.DataFrame,
    series: List[tuple],
    title: str,
    path: str,
    deltas: Optional[List[tuple]] = None,
    per_row: int = 20,
) -> None:
    """
    Grouped bars per dataset-level, one row of panels per dataset group (wrapped every
    `per_row` dataset-levels), all panels on the same scale.

    Args:
        table: One row per unit, with `{col}` and `{col}_ci` for every series column.
        series: (column, label, colour) of each bar.
        deltas: (column_a, column_b) pairs: b - a is written above the pair.
    """

    chunks = []
    for group in pd.unique(table["group"]):
        sub = table[table["group"] == group]
        for s in range(0, len(sub), per_row):
            chunks.append((group, sub.iloc[s : s + per_row]))
    n_max = max(len(c) for _, c in chunks)
    nb = len(series)
    width = 0.8 / nb
    fig, axes = plt.subplots(
        len(chunks), 1, figsize=(max(10.0, 2.0 + max(0.9, 0.3 * nb) * n_max), 3.4 * len(chunks)), squeeze=False
    )
    for ax, (group, sub) in zip(axes[:, 0], chunks):
        pos = np.arange(len(sub))
        for b, (col, label, color) in enumerate(series):
            ax.bar(
                pos + (b - (nb - 1) / 2) * width,
                sub[col],
                width,
                yerr=sub[f"{col}_ci"],
                color=color,
                capsize=2,
                error_kw=dict(lw=0.7, ecolor="#444"),
                label=label,
            )
        for a, b in deltas or []:
            ia = [s[0] for s in series].index(a)
            ib = [s[0] for s in series].index(b)
            xc = pos + ((ia + ib) / 2 - (nb - 1) / 2) * width
            top = np.nanmax(np.c_[sub[a] + sub[f"{a}_ci"], sub[b] + sub[f"{b}_ci"]], axis=1)
            for x_, t_, d in zip(xc, top, sub[b] - sub[a]):
                if np.isfinite(d):
                    ax.text(
                        x_,
                        min(t_ + 0.012, 1.0),
                        f"{d:+.3f}",
                        ha="center",
                        fontsize=6.5,
                        color="#1a7f37" if d >= 0 else "#c62828",
                        rotation=90 if nb > 2 else 0,
                    )
        ax.set_xticks(pos)
        ax.set_xticklabels([unit_label(r) for _, r in sub.iterrows()], fontsize=7, rotation=0)
        ax.set_xlim(-0.6, n_max - 0.4)
        acc_axis(ax, hi=1.05)
        ax.set_ylabel(f"{group}\nbalanced accuracy", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=nb, frameon=False, fontsize=9)
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    save(fig, path)


def confusion_plot(paths: List[str], title: str, path: str) -> None:
    """Row-normalised confusion (recall) matrices without / with PPSA and their difference."""

    cms = {"raw": 0, "ppsa": 0}
    for p in paths:
        with open(os.path.join(p, "metrics.json")) as f:
            conf = json.load(f)["confusion"]
        classes = conf["classes"]
        for k in cms:
            cms[k] = cms[k] + np.asarray(conf[k], dtype=float)
    rec = {k: v / np.maximum(v.sum(axis=1, keepdims=True), 1) for k, v in cms.items()}
    diff = rec["ppsa"] - rec["raw"]
    lim = max(float(np.abs(diff).max()), 0.05)
    n = len(classes)
    fs = 7 if n <= 8 else 5.5

    fig, axes = plt.subplots(1, 3, figsize=(3 * (2.2 + 0.42 * n), 1.4 + 0.42 * n))
    panels = [
        (rec["raw"], "without PPSA", "Blues", 0, 1, "fraction of the true class"),
        (rec["ppsa"], "with PPSA", "Blues", 0, 1, "fraction of the true class"),
        (diff, "difference (with - without PPSA)", "RdBu_r", -lim, lim, "difference in fraction of the true class"),
    ]
    for ax, (mat, name, cmap, lo, hi, cbar) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=hi)
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                dark = abs(v) > 0.6 * hi  # the colour maps are dark at their high end (and low end for the difference)
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fs, color="white" if dark else "black")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicted type")
        ax.set_ylabel("true type")
        ax.set_title(name, fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save(fig, path)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def feature_plots(sm: pd.DataFrame, runs: pd.DataFrame, keys: List[str], feature: str, cfg: dict):
    """All the figures of one feature type; returns the ranking tests and the selected combination."""

    out = os.path.join(cfg["plots_dir"], feature)
    n_seeds = int(sm["n_seeds"].max())
    all_tests = []

    # combinations, per scope and per dataset
    for variant, (metric, vlabel) in VARIANTS.items():
        for name, sub in scopes(sm, cfg.get("plots", {}).get("class_bins", [])).items():
            agg = scope_means(sub, keys, metric)
            what = scope_title(name, sub)
            combo_barplot(
                agg,
                keys,
                f"[{feature}] {what}, {vlabel}: mean balanced accuracy +- 95% CI across dataset-levels",
                os.path.join(out, "barplots", f"{name}_{variant}.png"),
                "bal. acc.",
            )
            ranking_plot(
                agg,
                f"[{feature}] best combinations by mean balanced accuracy, {what}, {vlabel}",
                os.path.join(out, "rankings", f"{name}_{variant}_balacc.png"),
                "mean balanced accuracy (+- 95% CI across dataset-levels)",
            )

            t, tests = rank_stats(sub, keys, metric)
            t.to_csv(os.path.join(cfg["results_dir"], f"ranking_{feature}_{name}_{variant}.csv"), index=False)
            all_tests.append({"feature": feature, "scope": name, "variant": variant, **tests})
            meanrank_plot(
                t,
                tests,
                f"[{feature}] best combinations by mean rank, {what}, {vlabel}",
                os.path.join(out, "rankings", f"{name}_{variant}_meanrank.png"),
            )
            compare_plot(
                t,
                tests,
                f"[{feature}] mean balanced accuracy vs mean rank, {what}, {vlabel}",
                os.path.join(out, "rankings", f"{name}_{variant}_compare.png"),
            )
        for (dataset, level), sub in sm.groupby(["dataset", "level"], sort=False):
            agg = sub.rename(columns={metric: "mean", f"{metric}_ci": "ci"})
            combo_barplot(
                agg,
                keys,
                f"[{feature}] {dataset} / {level} ({sub['n_classes'].iloc[0]} types), {vlabel}: "
                f"mean balanced accuracy +- 95% CI over {n_seeds} seeds",
                os.path.join(out, "per_dataset", f"{dataset}_{level}_{variant}.png"),
                "bal. acc.",
            )

    # the selected combination on every dataset: all the bars of a dataset come from its runs
    sel = select_combination(sm, keys, feature, cfg)
    combination = sel["combination"]
    t = sm[sm["combination"] == combination]
    t.to_csv(os.path.join(cfg["results_dir"], f"selected_{feature}.csv"), index=False)
    n_missing = sm.groupby(["dataset", "level"]).ngroups - len(t)
    if n_missing:
        print(f"{feature}: {combination} has no run on {n_missing} dataset-levels, left out of the figures")
    note = f"combination {combination.replace('_', ' ')} ({sel['chosen_by']}); error bars = 95% CI over {n_seeds} seeds"

    unit_barplot(
        t,
        [("bal_acc_raw", "without PPSA", LIGHT), ("bal_acc_ppsa", "with PPSA", DARK)],
        f"[{feature}] balanced accuracy without / with PPSA\n{note}",
        os.path.join(out, "ppsa_effect.png"),
        deltas=[("bal_acc_raw", "bal_acc_ppsa")],
    )

    unit_barplot(
        t,
        [
            ("bal_acc_raw", "no PPSA", LIGHT),
            ("bal_acc_ppsa", "PPSA everywhere", DARK),
            ("bal_acc_gated", "gated (inside spots only)", "#238b45"),
            ("bal_acc_antigated", "anti-gated (outside spots only)", "#cb181d"),
        ],
        f"[{feature}] where PPSA is applied\n{note}",
        os.path.join(out, "ppsa_gated.png"),
    )

    cols = ["bal_acc_raw_in_spot", "bal_acc_ppsa_in_spot", "bal_acc_raw_out_spot", "bal_acc_ppsa_out_spot"]
    unit_barplot(
        t,
        [
            (cols[0], "inside spots, no PPSA", "#c6dbef"),
            (cols[1], "inside spots, PPSA", "#08519c"),
            (cols[2], "outside spots, no PPSA", "#fdd0a2"),
            (cols[3], "outside spots, PPSA", "#d94801"),
        ],
        f"[{feature}] cells inside vs outside spots\n{note}",
        os.path.join(out, "ppsa_split.png"),
        deltas=[(cols[0], cols[1]), (cols[2], cols[3])],
    )

    cols = ["bal_acc_raw_train_spot", "bal_acc_ppsa_train_spot", "bal_acc_raw_test_spot", "bal_acc_ppsa_test_spot"]
    unit_barplot(
        t,
        [
            (cols[0], "train spots, no PPSA", "#c6dbef"),
            (cols[1], "train spots, PPSA", "#08519c"),
            (cols[2], "test spots, no PPSA", "#fdd0a2"),
            (cols[3], "test spots, PPSA", "#d94801"),
        ],
        f"[{feature}] cells of the training spots vs of the test spots\n{note}",
        os.path.join(out, "ppsa_traintest.png"),
        deltas=[(cols[0], cols[1]), (cols[2], cols[3])],
    )

    for _, b in t.iterrows():
        r = runs[
            (runs["dataset"] == b["dataset"]) & (runs["level"] == b["level"]) & (runs["combination"] == combination)
        ]
        title = (
            f"[{feature}] {b['dataset']} / {b['level']}, selected combination {combination.replace('_', ' ')} "
            f"({len(r)} seeds summed)\nbalanced accuracy {b['bal_acc_raw']:.3f} without PPSA, "
            f"{b['bal_acc_ppsa']:.3f} with PPSA; rows = true type, normalised per row (diagonal = recall)"
        )
        confusion_plot(list(r["path"]), title, os.path.join(out, "confusion", f"{b['dataset']}_{b['level']}.png"))
    return all_tests, sel


def features_plot(sm: pd.DataFrame, cfg: dict, selected: Dict[str, str]) -> None:
    """Feature types side by side on the datasets they share, each with its selected combination."""

    features = list(pd.unique(sm["feature"]))
    if len(features) < 2:
        return
    n_features = sm.groupby(["dataset", "level"], sort=False)["feature"].transform("nunique")
    sm = sm[n_features == len(features)]
    base = None
    series, palette = [], plt.cm.tab10.colors
    for i, f in enumerate(features):
        t = sm[(sm["feature"] == f) & (sm["combination"] == selected[f])]
        t = t[
            [
                "group",
                "dataset",
                "level",
                "n_classes",
                "bal_acc_raw",
                "bal_acc_raw_ci",
                "bal_acc_ppsa",
                "bal_acc_ppsa_ci",
            ]
        ]
        t = t.rename(columns={c: f"{f}:{c}" for c in t.columns if c.startswith("bal_acc")})
        base = t if base is None else base.merge(t, on=["group", "dataset", "level", "n_classes"])
        light = tuple(0.55 + 0.45 * np.array(palette[i]))
        series += [(f"{f}:bal_acc_raw", f"{f}, no PPSA", light), (f"{f}:bal_acc_ppsa", f"{f}, PPSA", palette[i])]
    n_seeds = int(sm["n_seeds"].max())
    combos = "; ".join(f"{f}: {selected[f].replace('_', ' ')}" for f in features)
    unit_barplot(
        base,
        series,
        f"Feature types compared, each with its selected combination ({combos})\n"
        f"error bars = 95% CI over {n_seeds} seeds",
        os.path.join(cfg["plots_dir"], "features.png"),
    )


def main(config: str, svg: bool = typer.Option(False, help="Also save every figure as SVG.")):
    global SAVE_SVG
    SAVE_SVG = svg
    cfg = load_config(config)
    runs_csv = os.path.join(cfg["results_dir"], "runs.csv")
    newest = max(
        (
            os.path.getmtime(p)
            for p in glob.glob(os.path.join(cfg["out_dir"], "*", "*", "*", "*", "seed_*", "metrics.json"))
        ),
        default=0,
    )
    if not os.path.exists(runs_csv) or os.path.getmtime(runs_csv) < newest:
        subprocess.run(
            [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "gridsearch.py"),
                "collect",
                config,
            ],
            check=True,
        )
    runs = pd.read_csv(runs_csv)
    keys = [k for k in cfg["grid"] if runs[k].nunique() > 1] if len(runs) else []
    sm = seed_means(runs, keys)
    order = {d["name"]: i for i, d in enumerate(cfg["datasets"])}
    sm = sm.sort_values(["dataset", "level"], key=lambda s: s.map(order) if s.name == "dataset" else s, kind="stable")
    tests, selected = [], []
    for feature in pd.unique(sm["feature"]):
        t, s = feature_plots(sm[sm["feature"] == feature], runs[runs["feature"] == feature], keys, feature, cfg)
        tests += t
        selected.append(s)
        print(f"{feature}: figures in {os.path.join(cfg['plots_dir'], feature)}")
    features_plot(sm, cfg, {s["feature"]: s["combination"] for s in selected})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(cfg["results_dir"], "ranking_tests.csv"), index=False)
    pd.DataFrame(selected).to_csv(os.path.join(cfg["results_dir"], "selected.csv"), index=False)
    for _, r in tests[tests["scope"] == "all"].iterrows():
        print(
            f"{r['feature']} {r['variant']}: elected {r['best']} (mean rank {r['best_mean_rank']:.1f}, "
            f"{r['n_tied_with_best']} other combinations not significantly worse; Friedman {fmt_p(r['friedman_p'])})"
        )
    for s in selected:
        print(
            f"{s['feature']}: selected {s['combination']} ({s['chosen_by']}), mean bal. acc. "
            f"{s['bal_acc_raw']:.3f} without PPSA, {s['bal_acc_ppsa']:.3f} with PPSA"
        )


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
    app.command()(main)
    app()
