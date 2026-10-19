"""
The slides: what the spots asked for, and what HEDeST put inside them.

One figure per (sample, level), three columns and two rows. The columns are the three things
being compared — the truth, HEDeST trained on the sample's own proportions, HEDeST trained on the
cell2location proportions — and the rows are the two scales:

* **spots**: one pie per spot. The first is the ground-truth composition; the other two are the
  composition of each spot recovered from the cells HEDeST assigned to it, so the three pies of a
  spot are directly comparable.
* **cells**: every segmented cell as a point, coloured by its type. The first panel is the true
  label, the other two the prediction. HEDeST labels the cells *outside* the spots as well, which
  is most of them, so this row shows far more than the spots do.

Predictions are the raw probabilities averaged over the ten seeds before the argmax, which is
what the aggregated prediction of a multi-seed run is; PPSA is not applied here.

The colours are the sample's own palette, the one its pseudo-Visium figures and the
deconvolution figures already use, so a type is the same colour everywhere in the study.

Usage: python slides.py [--config visiumhd.yaml] [--samples P2 P5] [--levels lv2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = HERE.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "gridsearch"))
sys.path.insert(0, str(STUDY / "deconv_code"))

import paths as D  # noqa: E402  (deconv_code/paths.py: the sample's files and palette)
from gridsearch import load_config  # type: ignore[attr-defined] # noqa: E402

FEATURE = "hoptimus0"
POINT_SIZE = 0.35


# ------------------------------------------------------------------------------------- the inputs


def unit_dir(cfg: dict, dataset: str, level: str) -> Path:
    return Path(cfg["out_dir"]) / FEATURE / dataset / level


def predicted_types(cfg: dict, dataset: str, level: str) -> pd.Series:
    """``{cell id: predicted type}``, from the seed-averaged raw probabilities."""

    unit = unit_dir(cfg, dataset, level)
    cells = [line.strip() for line in (unit / "cell_ids.txt").read_text().splitlines() if line.strip()]

    combinations = sorted(p for p in unit.glob("*") if p.is_dir())
    if len(combinations) != 1:
        raise SystemExit(f"{unit}: expected one combination, found {[p.name for p in combinations]}")

    stack, classes = [], None
    for seed_dir in sorted(combinations[0].glob("seed_*")):
        path = seed_dir / "predictions.npz"
        if not path.exists():
            continue
        payload = np.load(path, allow_pickle=True)
        classes = [str(name) for name in payload["classes"]]
        stack.append(payload["raw"].astype("float32"))
    if not stack:
        raise SystemExit(f"{combinations[0]}: no seed has predictions.npz")
    if stack[0].shape[0] != len(cells):
        raise SystemExit(f"{unit}: {stack[0].shape[0]} prediction rows for {len(cells)} cell ids")

    mean = np.mean(stack, axis=0)
    print(f"    {dataset}/{level}: {len(stack)} seeds averaged over {mean.shape[0]:,} cells")

    return pd.Series([classes[i] for i in mean.argmax(axis=1)], index=pd.Index(cells, name="cell_id"))


def composition(assignment: pd.Series, spot_dict: dict, vocabulary: list[str], spots: list[str]) -> pd.DataFrame:
    """Per spot, the share of its cells of each type — the spots' view of a cell assignment."""

    rows = {}
    for spot in spots:
        kinds = assignment.reindex([c for c in spot_dict.get(spot, [])]).dropna()
        counts = kinds.value_counts()
        total = float(counts.sum())
        rows[spot] = {k: float(counts.get(k, 0.0)) / total if total else 0.0 for k in vocabulary}

    return pd.DataFrame.from_dict(rows, orient="index")[vocabulary]


# ------------------------------------------------------------------------------------- the figure


def _pies(ax, coords, frame, colours, radius) -> None:
    """One pie per spot at its own position, as the deconvolution figures draw them."""

    wedges, shades = [], []
    for (x, y), row in zip(coords, frame.to_numpy(float)):
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


def figure(sample: D.Sample, level: str, cfg: dict, out: Path) -> None:
    """The six panels of one (sample, level)."""

    import anndata as ad

    vocabulary = sample.vocabulary(level)
    colours = sample.palette(level)
    truth = pd.read_csv(sample.truth(level), index_col=0)[vocabulary]

    spots = ad.read_h5ad(sample.spots)
    entry = next(iter(spots.uns["spatial"].values()))
    image = np.asarray(entry["images"]["hires"])
    factors = entry["scalefactors"]
    scale = float(factors["tissue_hires_scalef"])
    pitch = (
        float(factors["spot_diameter_fullres"])
        * scale
        * (spots.uns["pseudovisium"]["spot_spacing_um"] / spots.uns["pseudovisium"]["spot_diameter_um"])
    )
    radius = 0.40 * pitch

    order = [s for s in truth.index if s in spots.obs_names]
    index = pd.Index(spots.obs_names).get_indexer(order)
    coords = spots.obsm["spatial"][index, :2] * scale

    cells = pd.read_csv(sample.root / "pseudovisium" / "sim" / "cell_types.csv")
    cells["cell_id"] = cells.cell_id.astype(str)
    cells = cells.set_index("cell_id")
    spot_dict = json.loads((sample.root / "pseudovisium" / "spot_dict.json").read_text())

    assignments = {"truth": cells[level].astype(str)}
    compositions = {"truth": truth.loc[order]}
    for key, dataset in [("gt", f"{sample.name}_gt"), ("deconv", f"{sample.name}_deconv")]:
        assignments[key] = predicted_types(cfg, dataset, level)
        compositions[key] = composition(assignments[key], spot_dict, vocabulary, order)

    # The cells cover more of the slide than the spots do, so the crop follows the cells and
    # both rows share it.
    x = cells.x.to_numpy(float) * scale
    y = cells.y.to_numpy(float) * scale
    margin = 3 * radius
    x0, x1 = max(x.min() - margin, 0.0), min(x.max() + margin, image.shape[1])
    y0, y1 = max(y.min() - margin, 0.0), min(y.max() + margin, image.shape[0])

    titles = [
        "ground truth",
        "HEDeST — trained on true proportions",
        "HEDeST — trained on deconvolved proportions",
    ]
    # The panels are `set_aspect("equal")`, so their height follows from the crop; the figure has
    # to be tall enough for both rows plus their titles, or the lower titles land on the upper
    # images.
    aspect = (y1 - y0) / (x1 - x0)
    panel = 16.5 / 3
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 2 * panel * aspect + 2.2), squeeze=False)

    # The column says where the labels come from and the row says at which scale they are drawn,
    # so only the columns get titles: a title on the lower row would sit on the image above it,
    # which tight_layout cannot prevent for axes whose aspect is fixed.
    for col, key in enumerate(["truth", "gt", "deconv"]):
        top, bottom = axes[0][col], axes[1][col]

        top.imshow(image, origin="upper")
        _pies(top, coords, compositions[key], colours, radius)
        top.set_title(titles[col], fontsize=11)

        bottom.imshow(image, origin="upper")
        kinds = assignments[key].reindex(cells.index)
        for kind in vocabulary:
            mask = (kinds == kind).to_numpy()
            if mask.any():
                bottom.scatter(x[mask], y[mask], s=POINT_SIZE, c=colours[kind], linewidths=0, rasterized=True)
        unknown = int(kinds.isna().sum())
        if unknown:
            print(f"    warning: {titles[col]} leaves {unknown:,} cells unlabelled")

        for ax in (top, bottom):
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)  # keep imshow's downward y
            ax.set_aspect("equal")
            ax.axis("off")

    handles = [Patch(facecolor=colours[k], label=k) for k in vocabulary]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False, fontsize=9)
    fig.suptitle(
        f"{sample.name} {level} — {len(vocabulary)} cell types, {len(cells):,} cells, {len(order):,} spots",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.03, 0.05, 1, 0.96), h_pad=1.5)

    # Placed from the final axes positions, so the row labels follow whatever tight_layout did.
    for row, label in enumerate(["spot composition", "cell assignment"]):
        box = axes[row][0].get_position()
        fig.text(0.012, box.y0 + box.height / 2, label, rotation=90, va="center", ha="left", fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "visiumhd.yaml"))
    parser.add_argument("--samples", nargs="+", default=D.SAMPLES)
    parser.add_argument("--levels", nargs="+", default=D.LEVELS)
    arguments = parser.parse_args()

    cfg = load_config(arguments.config)
    out = Path(cfg["plots_dir"]) / "slides"

    for name in arguments.samples:
        sample = D.sample(name)
        for level in arguments.levels:
            print(f"  {name} {level}")
            figure(sample, level, cfg, out / f"slide_{name}_{level}.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
