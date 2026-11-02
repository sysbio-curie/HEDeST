"""
What each method costs: runtime and model size, in one figure.

Reads ``collect_timing.py``'s ``timing_per_run.csv`` and ``count_params.py``'s
``params_per_unit.csv`` and writes ``plots/cost_comparison.{png,svg}``.

Three panels, because the two quantities the comparison is about do not share an axis and
the composition of the runtime is the part that explains it:

    A  runtime per run, log scale, one column per method and GPU type. Every run is drawn,
       so the spread across sample-levels is visible rather than averaged away.
    B  what that runtime is spent on, as a share: image features, reading the inputs, and
       the model itself. Shares rather than seconds, so a method that takes twenty times
       longer can still be read beside one that does not.
    C  parameters, log scale, split into the weights that are fitted and the pretrained
       backbone that is only run.

Panel A is split by GPU type and labels it on every column, because HistoCell's own records
are 5.5x slower on this partition's P100 nodes than on its A40 node: a runtime quoted
without its hardware means very little here. In practice only P100 is drawn, since that is
the one type all three methods were measured on -- see ``collect_timing.py`` on why there
is no A40 row for HEDeST. The panel draws whichever types the table actually contains.

    python benchmark/comparison_code/plot_cost.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import config as C  # noqa: E402

TIMING_DIR = os.path.join(C.OUT_DIR, "timing")
ORDER = ["HEDeST", "HistoCell", "PanoSpace"]
GPUS = {"a40": "A40", "p100": "P100"}

# HEDeST is drawn in the strong blue it carries as the headline method elsewhere in this
# study; the other two keep their own colours.
COLOR = {
    "HEDeST": C.METHODS["HEDeST + PPSA"],
    "HistoCell": C.METHODS["HistoCell"],
    "PanoSpace": C.METHODS["PanoSpace"],
}

# The runtime is split three ways. A lighter shade of the method's colour per component,
# so a column in panel B still reads as belonging to its method.
COMPONENTS = [
    ("t_feature", "image features"),
    ("t_load", "reading the inputs"),
    ("t_model", "the model itself"),
]
SHADES = [1.0, 0.62, 0.40]


def save_fig(fig: plt.Figure, stem: str, dpi: int = 200) -> None:
    """Saves ``fig`` as ``stem.png`` and ``stem.svg`` and closes it."""

    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def shade(color: str, weight: float) -> tuple:
    """Mixes ``color`` with white; ``weight`` 1 is the colour itself, 0 is white."""

    rgb = np.asarray(matplotlib.colors.to_rgb(color))

    return tuple(weight * rgb + (1.0 - weight) * np.ones(3))


def _hours(seconds: float) -> str:
    """Formats a duration the way a reader thinks about it."""

    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"

    return f"{seconds / 3600:.1f} h"


def panel_runtime(ax: plt.Axes, df: pd.DataFrame) -> list:
    """Draws every run's total runtime, grouped by method and GPU type."""

    ticks, labels, slot = [], [], 0
    present = [g for g in GPUS if (df.gpu == g).any()]
    rng = np.random.default_rng(0)

    for method in ORDER:
        drawn = False
        for gpu in present:
            sub = df[(df.method == method) & (df.gpu == gpu)]
            if sub.empty:
                # a method measured on fewer GPU types than another leaves no blank column
                continue
            drawn = True
            values = sub.t_total.to_numpy()
            # a little horizontal jitter, otherwise 117 runs sit on one line
            ax.scatter(
                slot + rng.uniform(-0.17, 0.17, len(values)),
                values,
                s=7,
                color=shade(COLOR[method], 0.45),
                alpha=0.55,
                lw=0,
                zorder=3,
            )
            mean = values.mean()
            ax.scatter(
                slot,
                mean,
                marker="D",
                s=70,
                color=COLOR[method],
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
            )

            # The feature step is charged to every seed above. Where it is in fact computed
            # once and reused, show what the run then costs: for HEDeST that is the
            # difference between competing on runtime and not competing at all, and it
            # belongs on the axis rather than in a footnote.
            amortised = sub.t_total_amortised.dropna()
            low = amortised.mean() if len(amortised) else None

            # When reuse buys little the two markers land on top of each other, so the
            # labels are stacked instead of both sitting on the midline.
            close = low is not None and mean / low < 1.5
            ax.annotate(
                _hours(mean),
                (slot, mean),
                textcoords="offset points",
                xytext=(13, 7 if close else 0),
                fontsize=8.5,
                va="center",
                color="0.25",
            )
            if low is not None:
                ax.plot([slot, slot], [low, mean], color=COLOR[method], lw=1.0, ls=":", zorder=4)
                # An unfilled face, so that where reuse buys almost nothing and the two
                # markers coincide the filled one still shows through the ring rather than
                # being painted over.
                ax.scatter(
                    slot,
                    low,
                    marker="D",
                    s=95,
                    facecolors="none",
                    edgecolors=COLOR[method],
                    linewidths=1.6,
                    zorder=6,
                )
                ax.annotate(
                    f"{_hours(low)} reusing",
                    (slot, low),
                    textcoords="offset points",
                    xytext=(13, -7 if close else 0),
                    fontsize=8.5,
                    va="center",
                    color="0.45",
                )

            ticks.append(slot)
            labels.append(f"{method}\n{GPUS[gpu]}")
            slot += 1
        if drawn:
            slot += 0.6

    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("seconds per run (log scale)")
    missing = [m for m in ORDER if len(present) > 1 and df[df.method == m].gpu.nunique() < len(present)]
    caveat = f"  |  {', '.join(missing)} measured on P100 only" if missing else ""
    ax.set_title(
        "A. Image + proportions to cell types, per run\n"
        f"39 sample-levels x 3 seeds; segmentation and deconvolution excluded{caveat}",
        fontsize=10,
        loc="left",
    )
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return present


def panel_composition(ax: plt.Axes, df: pd.DataFrame, present: list) -> None:
    """Draws what the runtime is spent on, as a share of each method's total."""

    rows, labels = [], []
    for method in ORDER:
        for gpu in present:
            sub = df[(df.method == method) & (df.gpu == gpu)]
            if sub.empty:
                continue
            total = sub.t_total.mean()
            rows.append((method, [sub[c].mean() / total for c, _ in COMPONENTS]))
            labels.append(f"{method}\n{GPUS[gpu]}")

    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for i, (_, name) in enumerate(COMPONENTS):
        widths = np.array([r[1][i] for r in rows])
        colors = [shade(COLOR[m], SHADES[i]) for m, _ in rows]
        ax.barh(y, widths, left=left, color=colors, edgecolor="white", lw=0.8, height=0.62)
        for yi, (w, l) in enumerate(zip(widths, left)):
            if w > 0.08:
                ax.text(
                    l + w / 2,
                    yi,
                    f"{w * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if i == 0 else "0.2",
                )
        left += widths

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("share of the runtime")
    ax.set_title("B. Where the time goes", fontsize=10, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        Patch(facecolor=shade("#777777", SHADES[i]), edgecolor="white", label=name)
        for i, (_, name) in enumerate(COMPONENTS)
    ]
    # under the axis: the bars span the full width, so there is no free corner inside
    ax.legend(
        handles=handles,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        ncol=3,
        handlelength=1.4,
        columnspacing=1.4,
    )


def panel_params(ax: plt.Axes, params: pd.DataFrame) -> None:
    """Draws the fitted weights and the frozen backbone of each method."""

    summary = (
        params.groupby("method")
        .agg(trainable=("trainable", "mean"), frozen=("frozen", "first"), frozen_name=("frozen_name", "first"))
        .reindex(ORDER)
    )

    y = np.arange(len(summary))
    for yi, (method, row) in enumerate(summary.iterrows()):
        ax.barh(yi, row.trainable, color=COLOR[method], height=0.5, zorder=3)
        ax.text(
            row.trainable * 1.35,
            yi,
            f"{row.trainable / 1e6:.2f} M fitted",
            va="center",
            fontsize=8.5,
            color="0.25",
        )
        if row.frozen:
            ax.barh(
                yi,
                row.frozen,
                left=0,
                color=shade(COLOR[method], 0.22),
                height=0.5,
                zorder=2,
                hatch="///",
                edgecolor="white",
                lw=0.0,
            )
            ax.text(
                row.frozen * 1.35,
                yi - 0.26,
                f"{row.frozen / 1e6:.0f} M frozen",
                va="center",
                fontsize=8,
                color="0.45",
            )

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{m}\n{summary.loc[m, 'frozen_name'].split(' (')[0]}" for m in summary.index],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("parameters (log scale)")
    ax.set_xlim(1e5, 1e11)
    ax.set_title(
        "C. Model size: weights fitted per run, and the backbone only run",
        fontsize=10,
        loc="left",
    )
    ax.grid(axis="x", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.legend(
        handles=[
            Patch(facecolor="#777777", label="fitted on each run"),
            Patch(facecolor=shade("#777777", 0.22), hatch="///", edgecolor="white", label="pretrained, frozen"),
        ],
        fontsize=8,
        loc="lower right",
        frameon=False,
    )


def main() -> None:
    """Builds the cost figure from the collected tables."""

    timing = pd.read_csv(os.path.join(TIMING_DIR, "timing_per_run.csv"))
    params = pd.read_csv(os.path.join(TIMING_DIR, "params_per_unit.csv"))
    timing = timing[timing.gpu.isin(GPUS)]
    for col, _ in COMPONENTS:
        timing[col] = timing[col].fillna(0.0)

    fig = plt.figure(figsize=(13.5, 9.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.42, wspace=0.26)
    ax_runtime = fig.add_subplot(grid[0, :])
    ax_comp = fig.add_subplot(grid[1, 0])
    ax_params = fig.add_subplot(grid[1, 1])

    present = panel_runtime(ax_runtime, timing)
    panel_composition(ax_comp, timing, present)
    panel_params(ax_params, params)

    note = (
        "Filled diamonds charge the image features to every seed, as if each run started "
        "from the image: H-Optimus-0 for HEDeST, DINOv2 for PanoSpace. Open diamonds count "
        "the reuse each one actually gets.\n"
        "The reuse is not comparable: HEDeST embeds a slide once and every level and seed "
        "of it reads the same file, where PanoSpace's cache is keyed to the sub-spot grid "
        "and so is rebuilt for each level."
    )
    fig.text(0.012, -0.005, note, fontsize=8.5, color="0.35", va="top")
    fig.suptitle(
        "What each method costs on the STHELAR samples (ground-truth proportions)",
        fontsize=12.5,
        x=0.012,
        ha="left",
        y=0.985,
    )

    stem = os.path.join(C.PLOT_DIR, "cost_comparison")
    save_fig(fig, stem)
    print(f"-> {stem}.png / .svg")


if __name__ == "__main__":
    main()
