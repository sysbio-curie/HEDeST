"""Shared plotting / statistics helpers for the benchmark scripts.

Trimmed to what the benchmark code actually imports (``mhast_benchmark.py``);
the notebook-only helpers went with the notebooks.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

    Args:
        metrics_list: List of dictionaries containing metrics from each run.

    Returns:
        Tuple containing the mean and confidence intervals for each metric.
    """

    df_metrics = pd.DataFrame(metrics_list)
    mean_values = df_metrics.mean().to_dict()
    std_values = df_metrics.std()
    count_values = df_metrics.count()
    se_values = std_values / np.sqrt(count_values)
    ci_values = {f"{key} ci": 1.96 * se for key, se in se_values.items()}

    return mean_values, ci_values
