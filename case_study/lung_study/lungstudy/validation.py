"""
Checks of HEDeST that need no ground truth.

- **Held-out spots.** Every seed was trained on 80% of the spots; the 10% of test spots it
  never saw (rebuilt exactly as HEDeST split them) are predicted from their cells' morphology
  alone. The deconvolution-only alternative for a place without a measured spot is to
  interpolate the proportions of the neighbouring (training) spots.
- **HoVer-Net nucleus types.** An independent morphological classifier (PanNuke), trained
  without any transcriptomics, run on the same nuclei.
- **Levels.** The HEDeST models of different annotation levels are trained independently:
  a cell called B cell at lv3 should be lymphoid at lv2.
- **Nucleus shape** per predicted type, against what the proportions alone can infer
  (a regression of the mean nucleus area of each spot on its proportions).
"""
from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from lungstudy import cache
from lungstudy import paths
from lungstudy.data import SampleData
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

# ------------------------------------------------------------------ held-out spots


def interpolate_from_neighbours(
    data: SampleData, targets: List[str], sources: List[str], k: int = 6, power: float = 1.0
) -> pd.DataFrame:
    """
    The deconvolution-only guess for spots whose proportions are hidden: the
    inverse-distance-weighted mean of the proportions of the ``k`` nearest source spots.
    """

    sources = [s for s in sources if s in data.props.index]
    source_xy = data.spots.loc[sources, ["x", "y"]].to_numpy()
    target_xy = data.spots.loc[targets, ["x", "y"]].to_numpy()
    distance, index = cKDTree(source_xy).query(target_xy, k=k)
    weights = 1.0 / np.maximum(distance, 1.0) ** power
    weights /= weights.sum(axis=1, keepdims=True)
    values = data.props.loc[sources].to_numpy()
    guess = np.einsum("tk,tkc->tc", weights, values[index])

    return pd.DataFrame(guess, index=targets, columns=data.ct)


def heldout_predictions(data: SampleData, split: str = "test") -> pd.DataFrame:
    """
    Per seed, the composition of each held-out spot as predicted by HEDeST (mean of its cells'
    predictions, without PPSA) and by interpolation of the neighbouring training spots, with
    the deconvolution truth.

    Returns:
        A long frame: seed, spot, method ('HEDeST', 'interpolation', 'deconvolution'), cell types.
    """

    table = cache.seed_spots(data.name)
    frames = []
    for seed, group in table.groupby("seed"):
        heldout = [s for s in group.loc[group["split"] == split, "spot"] if s in data.props.index]
        train = list(group.loc[group["split"] == "train", "spot"])
        hedest = group.set_index("spot").loc[heldout, data.ct]
        interpolation = interpolate_from_neighbours(data, heldout, train)
        truth = data.props.loc[heldout, data.ct]
        for method, frame in (("HEDeST", hedest), ("interpolation", interpolation), ("deconvolution", truth)):
            frame = frame.copy()
            frame.insert(0, "method", method)
            frame.insert(0, "spot", frame.index)
            frame.insert(0, "seed", seed)
            frames.append(frame.reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)


def heldout_metrics(data: SampleData, split: str = "test") -> Dict[str, pd.DataFrame]:
    """
    Scores of HEDeST and interpolation against the deconvolution on held-out spots.

    Returns:
        ``per_type``: seed x method x cell type Pearson r and MAE;
        ``per_spot``: seed x spot x method total-variation distance (0 = identical, 1 = disjoint),
        with the local heterogeneity of the spot (its distance to its neighbours' mean).
    """

    long = heldout_predictions(data, split)
    per_type, per_spot = [], []
    for seed, group in long.groupby("seed"):
        truth = group[group["method"] == "deconvolution"].set_index("spot")[data.ct]
        for method in ("HEDeST", "interpolation"):
            guess = group[group["method"] == method].set_index("spot").loc[truth.index, data.ct]
            for cell_type in data.ct:
                x, y = truth[cell_type].to_numpy(), guess[cell_type].to_numpy()
                constant = x.std() == 0 or y.std() == 0
                per_type.append(
                    {
                        "seed": seed,
                        "method": method,
                        "cell_type": cell_type,
                        "pearson": np.nan if constant else pearsonr(x, y)[0],
                        "mae": float(np.abs(x - y).mean()),
                        "mean_truth": float(x.mean()),
                    }
                )
            tv = 0.5 * np.abs(truth.to_numpy() - guess.to_numpy()).sum(axis=1)
            per_spot.append(pd.DataFrame({"seed": seed, "spot": truth.index, "method": method, "tv": tv}))

    per_spot = pd.concat(per_spot, ignore_index=True)
    heterogeneity = per_spot[per_spot["method"] == "interpolation"].set_index(["seed", "spot"])["tv"]
    per_spot["heterogeneity"] = heterogeneity.reindex(pd.MultiIndex.from_frame(per_spot[["seed", "spot"]])).to_numpy()

    return {"per_type": pd.DataFrame(per_type), "per_spot": per_spot}


def neighbour_dispersion(data: SampleData, split: str = "test", k: int = 6) -> pd.DataFrame:
    """
    How heterogeneous the neighbourhood of each held-out spot is, from its neighbours only
    (the mean distance of the k nearest training spots to their own mean), so that the
    measure does not favour either method.
    """

    table = cache.seed_spots(data.name)
    rows = []
    for seed, group in table.groupby("seed"):
        train = [s for s in group.loc[group["split"] == "train", "spot"] if s in data.props.index]
        heldout = [s for s in group.loc[group["split"] == split, "spot"] if s in data.props.index]
        _, index = cKDTree(data.spots.loc[train, ["x", "y"]].to_numpy()).query(
            data.spots.loc[heldout, ["x", "y"]].to_numpy(), k=k
        )
        neighbours = data.props.loc[train].to_numpy()[index]
        dispersion = 0.5 * np.abs(neighbours - neighbours.mean(axis=1, keepdims=True)).sum(axis=2).mean(axis=1)
        rows.append(pd.DataFrame({"seed": seed, "spot": heldout, "dispersion": dispersion}))

    return pd.concat(rows, ignore_index=True)


def blend_curve(data: SampleData, weights=(0, 0.25, 0.5, 0.75, 1.0), split: str = "test") -> pd.DataFrame:
    """Error of w * HEDeST + (1 - w) * interpolation on held-out spots, for several w."""

    long = heldout_predictions(data, split)
    rows = []
    for seed, group in long.groupby("seed"):
        truth = group[group["method"] == "deconvolution"].set_index("spot")[data.ct]
        hedest = group[group["method"] == "HEDeST"].set_index("spot").loc[truth.index, data.ct]
        interpolation = group[group["method"] == "interpolation"].set_index("spot").loc[truth.index, data.ct]
        for w in weights:
            blend = w * hedest + (1 - w) * interpolation
            rows.append({"seed": seed, "w_hedest": w, "tv": float((0.5 * (blend - truth).abs().sum(axis=1)).mean())})

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ HoVer-Net types

# HoVer-Net's PanNuke classes, grouped into the three compartments both models can name.
HOVERNET_COMPARTMENT = {1: "epithelial", 5: "epithelial", 2: "immune", 3: "stromal"}
LV1_COMPARTMENT = {"epithelial": "epithelial", "immune": "immune", "stromal": "stromal", "endothelial": "stromal"}


def compartment_labels(name: str) -> pd.DataFrame:
    """
    Per cell, in three compartments (epithelial / immune / stromal): HoVer-Net's call, HEDeST's
    lv1 call, and the deconvolution-only call (the most abundant lv1 type of the cell's spot, or
    of the nearest spot for a cell between spots).
    """

    data = SampleData(name, "lv1")
    cells = data.cells
    dominant = data.props.idxmax(axis=1)
    spot_of = cells["spot"].where(cells["in_spot"], cells["nearest_spot"])
    table = pd.DataFrame(
        {
            "hovernet": cells["hv_type"].map(HOVERNET_COMPARTMENT),
            "hv_prob": cells["hv_prob"],
            "HEDeST": cells["label"].map(LV1_COMPARTMENT),
            "deconvolution": spot_of.map(dominant).map(LV1_COMPARTMENT),
            "in_spot": cells["in_spot"],
        },
        index=cells.index,
    )

    return table.dropna(subset=["hovernet"])


def agreement_scores(table: pd.DataFrame, methods=("HEDeST", "deconvolution")) -> pd.DataFrame:
    """Accuracy, Cohen's kappa and per-class recall against HoVer-Net, per method and location."""

    from sklearn.metrics import cohen_kappa_score

    rows = []
    for where, subset in (
        ("all cells", table),
        ("in spots", table[table["in_spot"]]),
        ("between spots", table[~table["in_spot"]]),
    ):
        subset = subset.dropna()
        for method in methods:
            row = {
                "where": where,
                "method": method,
                "n": len(subset),
                "accuracy": float((subset[method] == subset["hovernet"]).mean()),
                "kappa": float(cohen_kappa_score(subset["hovernet"], subset[method])),
            }
            for compartment in ("epithelial", "immune", "stromal"):
                selection = subset["hovernet"] == compartment
                row[f"recall_{compartment}"] = float((subset.loc[selection, method] == compartment).mean())
            rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ levels


# The annotation hierarchy, from the names of the cell2location levels. lv3 "B cell" holds the
# whole B lineage (lv4 B cell + plasma cell); mesothelial joins the stroma at lv1, the coarse
# type its proportions follow best.
HIERARCHY = {
    ("lv4", "lv3"): {"T CD4": "T cell", "T CD8": "T cell", "plasma cell": "B cell"},
    ("lv3", "lv2"): {
        "B cell": "immune_lymphoid",
        "T cell": "immune_lymphoid",
        "NK cell": "immune_lymphoid",
        "DC": "immune_myeloid",
        "macrophage": "immune_myeloid",
        "mast cell": "immune_myeloid",
        "CAF/fibro": "fibroblast",
        "SMC": "perivascular",
        "perictyes": "perivascular",
    },
    ("lv2", "lv1"): {
        "immune_lymphoid": "immune",
        "immune_myeloid": "immune",
        "fibroblast": "stromal",
        "perivascular": "stromal",
        "mesothelial": "stromal",
    },
}


def hierarchy(fine: str, coarse: str, names: List[str]) -> pd.Series:
    """The parent of each fine type (itself when the name is kept at the coarser level)."""

    rules = HIERARCHY[(fine, coarse)]
    return pd.Series({n: rules.get(n, n) for n in names}, name=f"parent_{coarse}")


def parent_mapping(names: List[str], fine: str, coarse: str) -> pd.Series:
    """
    The parent of each fine type at the coarse level, taken from the deconvolution itself: the
    coarse type whose proportions it follows best across the spots of all the samples.
    """

    frames_fine, frames_coarse = [], []
    for name in names:
        fine_props = cache.proportions(name, fine)
        coarse_props = cache.proportions(name, coarse)
        shared = fine_props.index.intersection(coarse_props.index)
        frames_fine.append(fine_props.loc[shared])
        frames_coarse.append(coarse_props.loc[shared])
    fine_props = pd.concat(frames_fine).fillna(0)
    coarse_props = pd.concat(frames_coarse).fillna(0)
    corr = pd.DataFrame(
        {
            c: [
                np.corrcoef(fine_props[f], coarse_props[c])[0, 1] if fine_props[f].std() > 0 else np.nan
                for f in fine_props
            ]
            for c in coarse_props
        },
        index=fine_props.columns,
    )

    return corr.idxmax(axis=1).rename(f"parent_{coarse}")


def level_consistency(name: str, mapping: pd.Series, fine: str, coarse: str) -> pd.DataFrame:
    """Per fine HEDeST type: how many of its cells the coarse HEDeST model puts in the parent type."""

    labels = cache.labels(name)
    fine_labels = labels[f"label_{fine}"]
    coarse_labels = labels[f"label_{coarse}"]
    expected = fine_labels.map(mapping)
    frame = pd.DataFrame({"fine": fine_labels, "coarse": coarse_labels, "consistent": coarse_labels == expected})
    out = frame.groupby("fine").agg(n=("consistent", "size"), consistent=("consistent", "mean"))
    out["parent"] = mapping.reindex(out.index).to_numpy()

    return out


# ------------------------------------------------------------------ nucleus shape


def area_from_proportions(data: SampleData, min_cells: int = 5, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    What the deconvolution alone can say about the nucleus size of each type: a non-negative
    regression of each spot's mean nucleus area on its proportions, with bootstrap intervals.
    HEDeST, by contrast, measures every nucleus of every type directly.
    """

    inside = data.cells[data.cells["in_spot"]]
    mean_area = inside.groupby("spot")["area_um2"].agg(["mean", "size"])
    mean_area = mean_area[mean_area["size"] >= min_cells]
    spots = mean_area.index.intersection(data.props.index)
    present = [c for c in data.ct if data.props.loc[spots, c].mean() > 0.01]
    X = data.props.loc[spots, present].to_numpy()
    y = mean_area.loc[spots, "mean"].to_numpy()
    coefficients, _ = nnls(X, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(y), len(y))
        boots.append(nnls(X[pick], y[pick])[0])
    boots = np.asarray(boots)

    return pd.DataFrame(
        {
            "deconvolution_estimate": coefficients,
            "low": np.quantile(boots, 0.025, axis=0),
            "high": np.quantile(boots, 0.975, axis=0),
        },
        index=pd.Index(present, name="cell_type"),
    )


def summarise(values: pd.Series, digits: int = 3) -> str:
    return f"{values.mean():.{digits}f} ± {values.std():.{digits}f}"


def maybe(value: Optional[float], fmt: str = ".3f") -> str:
    return "–" if value is None or not np.isfinite(value) else format(value, fmt)


__all__ = [
    "heldout_predictions",
    "heldout_metrics",
    "compartment_labels",
    "agreement_scores",
    "parent_mapping",
    "level_consistency",
    "area_from_proportions",
    "paths",
]
