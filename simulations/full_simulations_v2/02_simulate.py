"""Step 2 - build every dataset of config.DATASETS into sim_v2/ (same layout as sim/).

Per dataset tag:
  {tag}_spot_dict.json               spot id -> cell ids
  {tag}_prop.csv                     cluster proportions per spot
  {tag}_perturb_{s}_prop.csv         perturbed proportions (only where perturb=True)
  {tag}_gt.csv                       one-hot cluster of every cell present in a spot
  {tag}_emb_dict.pt                  H-Optimus-0 embeddings of those cells
  {tag}_image_dict_64px_20um.pt      64 px / 20 um cell images of those cells
Duplicated cells are named "<id>-1" and share the embedding / image of <id>.
Figures (png + svg) in figures/datasets/{tag}/.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from utils.data_simulation import add_perturbation  # noqa: E402
from utils.data_simulation import create_bags  # noqa: E402
from utils.data_simulation import duplicate_cluster  # noqa: E402
from utils.data_simulation import get_bag_proportions  # noqa: E402
from utils.data_simulation import plot_bags  # noqa: E402
from utils.data_simulation import plot_cluster_distribution  # noqa: E402
from utils.data_simulation import plot_cooccurrence  # noqa: E402
from utils.data_simulation import plot_perturbation  # noqa: E402
from utils.data_simulation import plot_spot_sizes  # noqa: E402
from utils.data_simulation import save_fig  # noqa: E402
from utils.data_simulation import sort_key  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_dataset(d: dict, cell_ids: np.ndarray, kmeans: dict, emb: dict, image_dict: dict) -> None:
    tag = C.dataset_tag(d)
    K = d["K"]
    out = lambda suffix: os.path.join(C.SIM_DIR, f"{tag}_{suffix}")  # noqa: E731
    fig_dir = os.path.join(C.FIG_DIR, "datasets", tag)
    log(f"== {tag}")

    ids = cell_ids[kmeans["pool_idx"]]
    labels = kmeans["pool_labels"]
    if d["dup"]:
        ids, labels = duplicate_cluster(ids, labels, C.DUPLICATED_CLUSTER)
    n_columns = K + int(d["dup"])
    not_mixed = [C.DUPLICATED_CLUSTER, K] if d["not_mixed"] else None

    bags, cells_drawn = create_bags(
        ids,
        labels,
        mean_cell_per_bag=d["mean"],
        var_cell_per_bag=d["var"],
        balance=C.dataset_weights(d),
        total_number_of_bags=d["n_spots"],
        random_state=C.SEED,
        not_mixed=not_mixed,
    )
    empty = [i for i, bag in enumerate(bags) if not bag]
    if empty:
        raise RuntimeError(f"{tag}: {len(empty)} empty spots (cells ran out before all spots were filled)")

    # proportions
    proportions = get_bag_proportions(bags, ids, labels)
    proportions.to_csv(out("prop.csv"), index=True)
    perturbed = {}
    if d["perturb"]:
        for i, strength in enumerate(C.PERTURBATION_STRENGTHS):
            perturbed[strength] = add_perturbation(proportions, strength=strength, random_state=[C.SEED, i])
            perturbed[strength].to_csv(out(f"perturb_{strength}_prop.csv"), index=True)

    # spot -> cells
    spot_dict = {str(spot_id): [str(c) for c in bag] for spot_id, bag in enumerate(bags)}
    with open(out("spot_dict.json"), "w") as f:
        json.dump(spot_dict, f)

    # cell-level files, restricted to the cells present in a spot
    sorted_cell_ids = sorted({c for bag in spot_dict.values() for c in bag}, key=sort_key)
    torch.save({c: image_dict[c.split("-")[0]].clone() for c in sorted_cell_ids}, out(f"{C.IMAGE_SUFFIX}.pt"))
    torch.save({c: emb[c.split("-")[0]].clone() for c in sorted_cell_ids}, out("emb_dict.pt"))

    id_to_cluster = dict(zip(ids, labels))
    columns = [f"Cluster {k}" for k in range(n_columns)]
    ground_truth = pd.DataFrame(0.0, index=pd.Index(sorted_cell_ids, name="nucleus_id"), columns=columns)
    for c in sorted_cell_ids:
        ground_truth.at[c, f"Cluster {id_to_cluster[c]}"] = 1.0
    ground_truth.to_csv(out("gt.csv"), index=True)

    # figures
    sizes = np.array([len(bag) for bag in bags])
    in_spots = ground_truth.values.sum(axis=0).astype(int)
    save_fig(plot_spot_sizes(sizes, d["mean"], d["var"], title=tag), os.path.join(fig_dir, "spot_sizes"))
    save_fig(
        plot_cluster_distribution(cells_drawn, in_spots, columns, title=tag),
        os.path.join(fig_dir, "cluster_distribution"),
    )
    save_fig(
        plot_bags(bags, ids, labels, title=tag, fig_edge_size=10 if d["n_spots"] <= 50 else 20, random_state=C.SEED),
        os.path.join(fig_dir, "bags"),
        dpi=120,
    )
    save_fig(
        plot_cooccurrence(bags, id_to_cluster, columns, title=f"{tag}\nspots containing both clusters"),
        os.path.join(fig_dir, "cooccurrence"),
    )
    if perturbed:
        save_fig(plot_perturbation(proportions, perturbed, title=tag), os.path.join(fig_dir, "perturbation"))

    log(
        f"   {len(bags)} spots, {len(sorted_cell_ids)} cells (drawn {int(cells_drawn.sum())}), spot size mean {sizes.mean():.2f} "
        f"var {sizes.var(ddof=1):.2f} | cells per cluster {in_spots.tolist()}"
    )


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    os.makedirs(C.SIM_DIR, exist_ok=True)

    cell_ids = np.load(os.path.join(C.CONSTRUCTION_DIR, "cell_ids.npy"))
    kmeans = {K: dict(np.load(os.path.join(C.CONSTRUCTION_DIR, f"kmeans_K{K}.npz"))) for K in C.KS}

    log(f"loading {C.EMB_PATH}")
    emb = torch.load(C.EMB_PATH, map_location="cpu", weights_only=False)
    assert list(emb.keys()) == cell_ids.tolist(), "embedding file changed since step 1"
    log(f"loading {C.IMAGE_DICT_PATH}")
    image_dict = torch.load(C.IMAGE_DICT_PATH, map_location="cpu", weights_only=False)

    for d in C.DATASETS:
        build_dataset(d, cell_ids, kmeans[d["K"]], emb, image_dict)

    log(f"step 2 done: {len(C.DATASETS)} datasets in {C.SIM_DIR}")


if __name__ == "__main__":
    main()
