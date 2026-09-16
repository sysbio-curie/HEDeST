"""Step 1 - cluster the H-Optimus-0 cell embeddings and plot the construction.

For every K in config.KS: K-means (Euclidean, raw 1536-d embeddings, n_init=10), then the
config.N_PER_CLUSTER cells closest to each centroid are retained. One K-means per K is
shared by every dataset with that K, so cluster numbers mean the same thing across them.

Cached in sim_v2/construction/ (reused on reruns, --force recomputes):
  cell_ids.npy, kmeans_K{K}.npz, umap.npy, centroids_xy.npy, clusters_summary.csv
Figures (png + svg) in figures/construction/K{K}/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from utils.data_simulation import find_closest_cells_to_clusters  # noqa: E402
from utils.data_simulation import perform_kmeans  # noqa: E402
from utils.data_simulation import perform_umap  # noqa: E402
from utils.data_simulation import plot_cells_per_cluster  # noqa: E402
from utils.data_simulation import plot_cluster_sizes  # noqa: E402
from utils.data_simulation import plot_slide_map  # noqa: E402
from utils.data_simulation import plot_umap  # noqa: E402
from utils.data_simulation import save_fig  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cached(name: str) -> str:
    return os.path.join(C.CONSTRUCTION_DIR, name)


def run_kmeans(X: np.ndarray, K: int, force: bool) -> dict:
    path = cached(f"kmeans_K{K}.npz")
    if os.path.exists(path) and not force:
        log(f"K={K}: using cached {path}")
        return dict(np.load(path))

    t0 = time.time()
    labels, distances, centroids = perform_kmeans(X, n_clusters=K, metric="euclidean", random_state=C.SEED)
    pool_idx = find_closest_cells_to_clusters(labels, distances, num_per_cluster=C.N_PER_CLUSTER)
    result = dict(
        labels=labels.astype(np.int32),
        centroids=centroids.astype(np.float32),
        dist_to_centroid=distances[np.arange(len(X)), labels].astype(np.float32),
        pool_idx=pool_idx.astype(np.int64),
        pool_labels=labels[pool_idx].astype(np.int32),
    )
    np.savez(path, **result)
    log(f"K={K}: K-means done in {time.time() - t0:.0f}s, cluster sizes {np.bincount(labels, minlength=K).tolist()}")
    return result


def run_umap(X: np.ndarray, force: bool) -> np.ndarray:
    path = cached("umap.npy")
    if os.path.exists(path) and not force:
        log(f"UMAP: using cached {path}")
        return np.load(path)

    t0 = time.time()
    Z = PCA(n_components=C.UMAP_PCA_DIM, random_state=C.SEED).fit_transform(X)
    xy = perform_umap(Z, n_neighbors=15, min_dist=0.1, random_state=C.SEED, verbose=True).astype(np.float32)
    np.save(path, xy)
    log(f"UMAP done in {time.time() - t0:.0f}s")
    return xy


def load_centroids(cell_ids: np.ndarray, force: bool) -> np.ndarray:
    path = cached("centroids_xy.npy")
    if os.path.exists(path) and not force:
        return np.load(path)

    with open(C.SEG_PATH) as f:
        nuc = json.load(f)["nuc"]
    xy = np.array([nuc[c]["centroid"] for c in cell_ids], dtype=np.float32)  # level-0 (x, y)
    np.save(path, xy)
    return xy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="recompute cached K-means / UMAP / centroids")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="notebook")
    os.makedirs(C.CONSTRUCTION_DIR, exist_ok=True)

    log(f"loading {C.EMB_PATH}")
    emb = torch.load(C.EMB_PATH, map_location="cpu", weights_only=False)
    cell_ids = np.array(list(emb.keys()))
    X = torch.stack(list(emb.values())).numpy()
    del emb
    log(f"{X.shape[0]} cells, dim {X.shape[1]}, dtype {X.dtype}")

    ids_path = cached("cell_ids.npy")
    if os.path.exists(ids_path) and not args.force:
        assert np.array_equal(np.load(ids_path), cell_ids), "cached cell ids differ from the embedding file"
    else:
        np.save(ids_path, cell_ids)

    results = {K: run_kmeans(X, K, args.force) for K in C.KS}

    rows = []
    for K, r in results.items():
        sizes = np.bincount(r["labels"], minlength=K)
        pool_sizes = np.bincount(r["pool_labels"], minlength=K)
        for k in range(K):
            d = r["dist_to_centroid"][r["pool_idx"][r["pool_labels"] == k]]
            rows.append(dict(K=K, cluster=k, n_cells=sizes[k], n_retained=pool_sizes[k], max_dist_retained=d.max()))
        if (sizes < C.N_PER_CLUSTER).any():
            log(
                f"WARNING K={K}: clusters with fewer than {C.N_PER_CLUSTER} cells: {np.where(sizes < C.N_PER_CLUSTER)[0].tolist()}"
            )
    pd.DataFrame(rows).to_csv(cached("clusters_summary.csv"), index=False)

    umap_xy = run_umap(X, args.force)
    del X

    centroids = load_centroids(cell_ids, args.force)
    scale = json.load(open(C.SCALEFACTORS_PATH))["tissue_hires_scalef"]  # level-0 px -> hires px
    background = np.asarray(Image.open(C.HIRES_IMAGE_PATH).convert("RGB"))

    log(f"loading {C.IMAGE_DICT_PATH}")
    image_dict = torch.load(C.IMAGE_DICT_PATH, map_location="cpu", weights_only=False)

    for K, r in results.items():
        t0 = time.time()
        out = os.path.join(C.FIG_DIR, "construction", f"K{K}")
        labels, pool_idx, pool_labels = r["labels"], r["pool_idx"], r["pool_labels"]
        pool_mask = np.zeros(len(cell_ids), dtype=bool)
        pool_mask[pool_idx] = True

        fig = plot_umap(
            umap_xy, labels, pool_mask, title=f"H-Optimus-0 cell embeddings, K-means K={K}", random_state=C.SEED
        )
        save_fig(fig, os.path.join(out, f"umap_K{K}"))

        fig = plot_cluster_sizes(
            np.bincount(labels, minlength=K),
            np.bincount(pool_labels, minlength=K),
            title=f"K-means K={K}: cluster sizes",
        )
        save_fig(fig, os.path.join(out, f"cluster_sizes_K{K}"))

        fig = plot_slide_map(
            background, centroids[pool_idx] * scale, pool_labels, K, title=f"Retained cells on the slide, K={K}"
        )
        save_fig(fig, os.path.join(out, f"slide_map_K{K}"))

        ncols = K if K <= 4 else (3 if K <= 6 else 5)
        nrows = int(np.ceil(K / ncols))
        for selection, name in (("top", "closest"), ("random", "random")):
            what = "64 cells closest to the centroid" if selection == "top" else "64 random retained cells"
            fig = plot_cells_per_cluster(
                image_dict,
                cell_ids[pool_idx],
                pool_labels,
                selection=selection,
                nrows=nrows,
                ncols=ncols,
                random_state=C.SEED,
                title=f"K={K}: {what}",
            )
            save_fig(fig, os.path.join(out, f"gallery_K{K}_{name}"), dpi=150)
        log(f"K={K}: figures done in {time.time() - t0:.0f}s")

    log("step 1 done")


if __name__ == "__main__":
    main()
