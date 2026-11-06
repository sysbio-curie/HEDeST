"""Shared plotting / statistics helpers for the benchmark scripts.

Trimmed to what the benchmark code actually imports (``mhast_benchmark.py``);
the notebook-only helpers went with the notebooks.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def bar_plot_perf(
    file_infos: list[tuple[str, str, str, str]],
    level: str = "cells",
    title: str = "",
    figsize: tuple = (12, 6),
    savefig: str = None,
    context: str = "talk",
) -> None:
    """
    Creates a Seaborn bar plot with error bars for performance metrics across models,
    supporting custom colors.

    Args:
    - file_infos: List of tuples (file_path, sheet_name, model_name, color)
    - level: 'cells' or 'spots'
    - title: Plot title
    - figsize: Tuple specifying figure size
    - savefig: Path to save figure, if desired
    - context: Seaborn context ('paper', 'notebook', 'talk', or 'poster')
    """

    # Set style and context
    sns.set_context(context)
    sns.set_style("whitegrid")

    # Determine metrics to extract
    if level == "cells":
        metrics = {
            # "Global Accuracy": "Global Acc.",
            "Balanced Accuracy": "Balanced Acc.",
            "Weighted F1 Score": "Weighted F1",
            "Weighted Precision": "Weighted Pre.",
            "Weighted Recall": "Weighted Rec.",
        }
    elif level == "spots":
        metrics = {"Pearson Correlation global": "Pearson Corr.", "Spearman Correlation global": "Spearman Corr."}
    else:
        raise ValueError("Level must be either 'cells' or 'spots'.")

    # Collect data and model colors
    df_rows = []
    model_colors = {}

    for file_info in file_infos:
        if len(file_info) == 4:
            file_path, sheet_name, model_name, color = file_info
            model_colors[model_name] = color
        else:
            file_path, sheet_name, model_name = file_info
            model_colors[model_name] = None  # fallback

        df = pd.read_excel(file_path, sheet_name=sheet_name).iloc[0]
        for long_name, short_name in metrics.items():
            df_rows.append(
                {
                    "Model": model_name,
                    "Metric": short_name,  # what will appear on x-axis
                    "Value": df[long_name],  # read from file using full name
                    "CI": df.get(f"{long_name} ci", 0),
                }
            )

    plot_df = pd.DataFrame(df_rows)

    # Create palette mapping
    palette = {model: color for model, color in model_colors.items() if color is not None}

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=plot_df, x="Metric", y="Value", hue="Model", palette=palette, errorbar=None, order=list(metrics.values())
    )

    # Add manual confidence intervals
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        y = row["Value"]
        ci = row["CI"]
        ax.errorbar(x=x, y=y, yerr=ci, fmt="none", c="black", capsize=5, lw=1.3)

    # Aesthetic adjustments
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.legend(title="", loc="lower right", frameon=True)
    plt.grid(True)
    plt.tight_layout()

    # Save if needed
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {savefig}")

    plt.show()


def compute_statistics(metrics_list: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """
    Computes mean and confidence intervals for a list of metrics.

    The confidence interval is the t-based 95% CI of the mean over the runs,
    ``t(0.975, n - 1) * sd / sqrt(n)``, as everywhere else in the benchmarks.

    Args:
        metrics_list: List of dictionaries containing metrics from each run.

    Returns:
        Tuple containing the mean and the CI half-width for each metric.
    """

    df_metrics = pd.DataFrame(metrics_list)
    mean_values = df_metrics.mean().to_dict()
    std_values = df_metrics.std(ddof=1)
    count_values = df_metrics.count()
    se_values = std_values / np.sqrt(count_values)
    t_values = stats.t.ppf(0.975, (count_values - 1).clip(lower=1))
    ci_values = {
        f"{key} ci": float(t * se) if n > 1 else 0.0
        for (key, se), t, n in zip(se_values.items(), t_values, count_values)
    }

    return mean_values, ci_values


def paired_t_test(a: pd.Series, b: pd.Series) -> dict[str, float]:
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
