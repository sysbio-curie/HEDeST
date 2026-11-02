"""
Held-out spots, used by the pre-check notebooks.

Every seed was trained on 80% of the spots; the 10% of test spots it never saw (rebuilt
exactly as HEDeST split them) are predicted from their cells' morphology alone. The
deconvolution-only alternative for a place without a measured spot is to interpolate the
proportions of the neighbouring (training) spots.
"""
from __future__ import annotations

from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from lungstudy import cache
from lungstudy.data import SampleData
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
