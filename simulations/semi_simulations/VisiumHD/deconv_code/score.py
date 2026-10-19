"""
Score the deconvolution against the ground truth and draw the comparison figures.

Metrics, per (sample, level):

  mean_pearson   per cell type, Pearson r between predicted and true proportion **across spots**,
                 then averaged over cell types unweighted — a rare type counts as much as a
                 dominant one, which is why rare types pull the mean down
  mae, rmse      elementwise over the spot x cell-type matrix
  mean_jsd       Jensen-Shannon divergence per spot, averaged over spots
  dominant_acc   share of spots whose argmax type is the true argmax type

The proportions scored are the ones ``aggregate.py`` wrote, so nothing is selected or copied here:
this reads results and reports them.

Colours come from each sample's own ``metadata.json``, the palette its pseudo-Visium figures
already use, so these sit beside those rather than inventing a second colour language.

    python score.py              # everything: metrics and figures
    python score.py --no-figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

PREDICTED_COLOUR = "#d08b2f"


# ------------------------------------------------------------------------------------- the inputs


def truth(sample: paths.Sample, level: str) -> pd.DataFrame:
    """The ground-truth proportions, in the vocabulary order the level declares."""

    frame = pd.read_csv(sample.truth(level), index_col=0)

    return frame[sample.vocabulary(level)]


def predicted(sample: paths.Sample, level: str) -> pd.DataFrame | None:
    """The deconvolved proportions, aligned to the level's vocabulary."""

    path = paths.proportions_path(sample.name, level)
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0)

    return frame.reindex(columns=sample.vocabulary(level)).fillna(0.0)


# ------------------------------------------------------------------------------------- the metrics


def jensen_shannon(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Jensen-Shannon divergence between two compositions (natural log, so <= ln 2)."""

    a = a / np.clip(a.sum(axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(b.sum(axis=1, keepdims=True), 1e-12, None)
    m = 0.5 * (a + b)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(a > 0, a * np.log(a / m), 0.0)
        right = np.where(b > 0, b * np.log(b / m), 0.0)

    return 0.5 * (left.sum(axis=1) + right.sum(axis=1))


def score_one(true: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Every metric for one (sample, level), plus the per-type correlations."""

    pred = pred.reindex(index=true.index).fillna(0.0)
    t, p = true.to_numpy(float), pred.to_numpy(float)

    per_type = {}
    for i, kind in enumerate(true.columns):
        # A type that is constant in either vector has no correlation to report at all, so it is
        # recorded as nan and left out of the mean rather than counted as a zero, which would
        # punish the deconvolution for something the data cannot measure. The per-type table keeps
        # the nan visible.
        if t[:, i].std() < 1e-12 or p[:, i].std() < 1e-12:
            per_type[kind] = float("nan")
        else:
            per_type[kind] = float(np.corrcoef(t[:, i], p[:, i])[0, 1])

    finite = [v for v in per_type.values() if np.isfinite(v)]

    return {
        "mean_pearson": float(np.mean(finite)) if finite else float("nan"),
        "mae": float(np.abs(t - p).mean()),
        "rmse": float(np.sqrt(((t - p) ** 2).mean())),
        "mean_jsd": float(jensen_shannon(t, p).mean()),
        "dominant_acc": float((t.argmax(axis=1) == p.argmax(axis=1)).mean()),
        "n_spots": int(len(true)),
        "n_types": int(true.shape[1]),
        "per_type_pearson": per_type,
    }


# ------------------------------------------------------------------------------------- the figures


def _pies(ax, coords, frame, colours, radius, title):
    """One pie per spot at its own position, as the pseudo-Visium overview figures draw them."""

    wedges, shades = [], []
    values = frame.to_numpy(float)
    for (x, y), row in zip(coords, values):
        total = row.sum()
        if total <= 0:
            continue
        start = 90.0
        for kind, share in zip(frame.columns, row / total):
            if share <= 0:
                continue
            extent = 360.0 * share
            wedges.append(Wedge((x, y), radius, start - extent, start))
            shades.append(colours[kind])
            start -= extent
    ax.add_collection(PatchCollection(wedges, facecolors=shades, edgecolors="none"))
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")


def figure_pies(sample: paths.Sample, level: str, true, pred, spots) -> None:
    """The slide twice, ground truth beside the deconvolution."""

    colours = sample.palette(level)
    entry = next(iter(spots.uns["spatial"].values()))
    image = np.asarray(entry["images"]["hires"])
    factors = entry["scalefactors"]
    scale = float(factors["tissue_hires_scalef"])

    # obsm["spatial"] is in full-resolution capture pixels, so both the positions and the spot
    # diameter scale into the hires image by the same factor. The spots are 55 um across on a
    # 100 um grid, which leaves a lot of background between pies; 0.40 of the pitch keeps them
    # clearly separated while staying large enough to read.
    order = [s for s in true.index if s in spots.obs_names]
    index = pd.Index(spots.obs_names).get_indexer(order)
    coords = spots.obsm["spatial"][index, :2] * scale
    pitch = (
        float(factors["spot_diameter_fullres"])
        * scale
        * (spots.uns["pseudovisium"]["spot_spacing_um"] / spots.uns["pseudovisium"]["spot_diameter_um"])
    )
    radius = 0.40 * pitch

    # Crop to the spots. The capture image is a lot wider than the tissue, and the empty margin
    # costs legibility on a figure whose point is the per-spot detail.
    margin = 3 * radius
    x0, x1 = coords[:, 0].min() - margin, coords[:, 0].max() + margin
    y0, y1 = coords[:, 1].min() - margin, coords[:, 1].max() + margin
    x0, y0 = max(x0, 0.0), max(y0, 0.0)
    x1, y1 = min(x1, image.shape[1]), min(y1, image.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    for ax, frame, title in (
        (axes[0], true.loc[order], f"{sample.name} {level} — ground truth"),
        (axes[1], pred.loc[order], f"{sample.name} {level} — cell2location"),
    ):
        ax.imshow(image, origin="upper")
        _pies(ax, coords, frame, colours, radius, title)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)  # keep imshow's downward y

    handles = [Patch(facecolor=colours[k], label=k) for k in true.columns]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(paths.figure_path(f"pies_{sample.name}_{level}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_scatter(sample: paths.Sample, level: str, true, pred) -> None:
    """Predicted against true proportion, one panel per cell type."""

    colours = sample.palette(level)
    kinds = list(true.columns)
    cols = min(4, len(kinds))
    rows = int(np.ceil(len(kinds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)

    for ax, kind in zip(axes.ravel(), kinds):
        x, y = true[kind].to_numpy(float), pred.loc[true.index, kind].to_numpy(float)
        r = np.corrcoef(x, y)[0, 1] if x.std() > 1e-12 and y.std() > 1e-12 else float("nan")
        ax.scatter(x, y, s=5, alpha=0.35, color=colours[kind])
        top = max(float(x.max()), float(y.max()), 1e-3)
        ax.plot([0, top], [0, top], lw=0.8, color="k", ls="--")
        ax.set_title(f"{kind}   r={r:.2f}", fontsize=10)
        ax.set_xlabel("true")
        ax.set_ylabel("cell2location")
    for ax in axes.ravel()[len(kinds) :]:
        ax.axis("off")

    fig.suptitle(f"{sample.name} {level} — per cell type, across {len(true):,} spots", fontsize=12)
    fig.tight_layout()
    fig.savefig(paths.figure_path(f"scatter_{sample.name}_{level}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_composition(sample: paths.Sample, level: str, true, pred) -> None:
    """Mean proportion per type: the truth against the deconvolution."""

    kinds = list(true.columns)
    frame = pd.DataFrame({"truth": true.mean(), "cell2location": pred.loc[true.index].mean()}).reindex(kinds)

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(kinds)), 4.2))
    frame.plot.bar(ax=ax, width=0.8, color={"truth": "0.2", "cell2location": PREDICTED_COLOUR})
    ax.set_ylabel("mean proportion over spots")
    ax.set_title(f"{sample.name} {level} — mean composition")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(paths.figure_path(f"composition_{sample.name}_{level}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_overview(metrics: dict, samples: dict) -> None:
    """Per-type Pearson for every sample and level, one panel each."""

    if not metrics:
        return

    names = sorted({key.split("/")[0] for key in metrics})
    levels = [lv for lv in paths.LEVELS if any(key.endswith(f"/{lv}") for key in metrics)]
    fig, axes = plt.subplots(len(names), len(levels), figsize=(4.8 * len(levels), 4.0 * len(names)), squeeze=False)

    for row, name in enumerate(names):
        for col, level in enumerate(levels):
            ax = axes[row][col]
            entry = metrics.get(f"{name}/{level}")
            if entry is None:
                ax.axis("off")
                continue

            # Worst at the top: the question this figure answers is which types the deconvolution
            # struggles with, and barh fills from the bottom up.
            series = pd.Series(entry["per_type_pearson"]).sort_values(ascending=True)
            palette = samples[name].palette(level)
            values = series.to_numpy(float)
            ax.barh(
                range(len(series)),
                np.nan_to_num(values),
                color=[palette.get(k, "0.6") for k in series.index],
                edgecolor="0.3",
                linewidth=0.4,
            )
            ax.axvline(entry["mean_pearson"], color="k", ls="--", lw=0.9)
            ax.set_yticks(range(len(series)), series.index, fontsize=7)
            ax.set_xlim(min(-0.05, float(np.nanmin(values)) - 0.05), 1.0)
            ax.set_xlabel("Pearson r across spots", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.set_title(
                f"{name} {level} — mean r {entry['mean_pearson']:.3f}, "
                f"MAE {entry['mae']:.3f}, JSD {entry['mean_jsd']:.3f}",
                fontsize=9,
            )
            for i, value in enumerate(values):
                # A type the data cannot correlate is nan: label it, but anchor it at the axis
                # rather than at nan, which would place nothing at all.
                anchor = max(value, 0.0) + 0.02 if np.isfinite(value) else 0.02
                ax.text(anchor, i, f"{value:.2f}" if np.isfinite(value) else "n/a", va="center", fontsize=6)

    fig.suptitle("cell2location vs ground truth — per cell type (dashed line: unweighted mean)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(paths.figure_path("metrics_overview"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------- driver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=paths.SAMPLES)
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args()

    metrics, objects = {}, {}
    for name in arguments.samples:
        sample = paths.sample(name)
        objects[name] = sample
        spots = ad.read_h5ad(sample.spots) if not arguments.no_figures else None

        for level in paths.LEVELS:
            pred = predicted(sample, level)
            if pred is None:
                print(f"  {name} {level}: no proportions yet, skipped")
                continue

            true = truth(sample, level)
            entry = score_one(true, pred)
            metrics[f"{name}/{level}"] = entry
            print(
                f"  {name} {level}: r={entry['mean_pearson']:.3f} mae={entry['mae']:.4f} "
                f"jsd={entry['mean_jsd']:.4f} dom={entry['dominant_acc']:.3f}"
            )

            if not arguments.no_figures:
                figure_pies(sample, level, true, pred.loc[true.index], spots)
                figure_scatter(sample, level, true, pred)
                figure_composition(sample, level, true, pred)

    paths.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "configuration": {
            "reference": f"reference_{paths.REFERENCE}.h5ad",
            "detection_alpha": paths.DETECTION_ALPHA,
            "reference_epochs": paths.REFERENCE_EPOCHS,
            "spatial_epochs": paths.SPATIAL_EPOCHS,
        },
        "metrics": metrics,
    }
    (paths.RESULT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {paths.RESULT_DIR / 'metrics.json'}")

    if not arguments.no_figures:
        figure_overview(metrics, objects)
        print(f"figures in {paths.FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
