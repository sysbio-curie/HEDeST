"""Single source of truth for the v2 fully simulated datasets (paths, seeds, weights, dataset list)."""
from __future__ import annotations

import os
from typing import Dict
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma"

# inputs
EMB_PATH = os.path.join(DATA, "hoptimus_tile_embed_OC.pt")  # tile-level H-Optimus-0, {cell_id: float32 (1536,)}
IMAGE_DICT_PATH = os.path.join(DATA, "image_dict_64px_20um.pt")  # {cell_id: uint8 (3, 64, 64)}, same cell order
SEG_PATH = os.path.join(DATA, "seg_json", "pannuke_fast_mask_lvl3.json")  # HoVerNet, centroids for the slide map
HIRES_IMAGE_PATH = os.path.join(DATA, "ST", "spatial", "tissue_hires_image.png")
SCALEFACTORS_PATH = os.path.join(DATA, "ST", "spatial", "scalefactors_json.json")

# outputs
SIM_DIR = os.path.join(DATA, "sim_v2")
CONSTRUCTION_DIR = os.path.join(SIM_DIR, "construction")  # clustering cache
FIG_DIR = os.path.join(HERE, "figures")

EMB_NAME = "hoptimus"  # replaces "moco" in the legacy tags
IMAGE_SUFFIX = "image_dict_64px_20um"

SEED = 42
KS = [3, 4, 6, 10, 15, 20]  # one K-means per K, shared by every dataset with that K
N_PER_CLUSTER = 2000  # cells closest to each centroid that are retained
DUPLICATED_CLUSTER = 0  # dup datasets: copy of this cluster becomes cluster K
PERTURBATION_STRENGTHS = [0.05, 0.1, 0.25, 0.5]
UMAP_PCA_DIM = 50  # UMAP is for display only; K-means runs on the raw embeddings

# Imbalanced cluster weights (normalised inside create_bags).
# K=4 and K=6 are the legacy weights recovered from sim/ (also in the article's Methods).
# K=3, 10, 15, 20 were drawn with utils.data_simulation.draw_imbalanced_weights(K, seed=42)
# and are hardcoded so the datasets do not depend on the numpy version.
IMBALANCED_WEIGHTS: Dict[int, List[float]] = {
    3: [0.6, 0.03899877659553103, 0.36100122340446905],
    4: [4.5, 0.5, 1, 2],
    6: [3.5, 0.5, 22, 2, 0.5, 4.5],
    10: [
        0.0682595678479546, 0.09330942164201787, 0.01233090499895755, 0.4807132374571908, 0.055222397393841195,
        0.09333178780594822, 0.01410751982002744, 0.08642988175000135, 0.08023599702369105, 0.016059284260369944,
    ],
    15: [
        0.4003222940370496, 0.0316879700253465, 0.03433820826485181, 0.07551482397073167, 0.040517436238656016,
        0.03193601007773949, 0.04234041797821476, 0.07409014595237273, 0.010990050430845418, 0.010404495860905517,
        0.08289017441905709, 0.014928369951541013, 0.06477066463387426, 0.011629130661568439, 0.07363980749724572,
    ],
    20: [
        0.06233219517050756, 0.03567406901530611, 0.3587389228121289, 0.023414036022486706, 0.04641516805846187,
        0.0624734521626104, 0.01100147527785789, 0.056701194499738435, 0.012545142002016564, 0.03538911391499301,
        0.009904745674124254, 0.03781688839496238, 0.022825158918318707, 0.044423569085645716, 0.02949219641700473,
        0.009904745674124254, 0.024957770407372806, 0.055152561178212316, 0.009904745674124254, 0.05093284964000333,
    ],
}  # fmt: skip
DRAWN_WEIGHT_KS = [3, 10, 15, 20]
DUPLICATE_WEIGHT = 2  # weight of the copy in imbalanced dup datasets (legacy [3.5, 0.5, 22, 2, 0.5, 4.5, 2])


def _dataset(K, n_spots, mean, var, balance, dup=False, not_mixed=False, perturb=False) -> dict:
    return dict(
        K=K, n_spots=n_spots, mean=mean, var=var, balance=balance, dup=dup, not_mixed=not_mixed, perturb=perturb
    )


DATASETS: List[dict] = [
    # same set of datasets as sim/
    _dataset(4, 30, 5, 5, "balanced"),
    _dataset(4, 30, 5, 5, "imbalanced"),
    _dataset(6, 200, 15, 15, "balanced", perturb=True),
    _dataset(6, 200, 15, 15, "imbalanced", perturb=True),
    _dataset(6, 200, 15, 15, "balanced", dup=True, perturb=True),
    _dataset(6, 200, 15, 15, "imbalanced", dup=True, perturb=True),
    _dataset(6, 200, 15, 15, "balanced", dup=True, not_mixed=True),
    _dataset(6, 200, 15, 15, "imbalanced", dup=True, not_mixed=True),
    # number-of-clusters series (K=6 of the series = the two 6-cluster datasets above)
    *[_dataset(K, 200, 15, 15, balance) for balance in ("balanced", "imbalanced") for K in (3, 10, 15, 20)],
]


def dataset_tag(d: dict) -> str:
    tag = f"{d['K']}_{EMB_NAME}_clusters_{d['n_spots']}spots_{d['balance']}_{d['mean']}mean_{d['var']}var"
    if d["dup"]:
        tag += "_dup"
    if d["not_mixed"]:
        tag += "_not_mixed"
    return tag


def dataset_weights(d: dict) -> List[float]:
    """Cluster weights of a dataset, one per proportion column (K, or K+1 with the duplicate)."""

    n_columns = d["K"] + int(d["dup"])
    if d["balance"] == "balanced":
        return [1.0] * n_columns
    return list(IMBALANCED_WEIGHTS[d["K"]]) + ([DUPLICATE_WEIGHT] if d["dup"] else [])
