"""Helpers for the fully simulated datasets (v2).

Copied from ``simulations/full_simulations/dataset_construction/utils/data_simulation.py``.
Changes to the copied functions do not alter their results, except where stated:

* ``create_bags``: the ``not_mixed`` check looks clusters up in a dict instead of scanning
  the whole id array for every cell (same result, much faster). It now raises when a
  cluster is asked for more cells than it holds, where the legacy code silently sampled
  cells with replacement.
* ``add_perturbation``: takes a ``random_state`` (the legacy noise was unseeded).
* ``get_bag_proportions``: starts from a zero-filled float frame (same values, no pandas
  downcasting warning).
* plotting functions save png + svg through ``save_fig`` instead of calling ``plt.show()``,
  and accept a ``random_state`` wherever they draw random numbers.

New: ``sort_key``, ``duplicate_cluster``, ``draw_imbalanced_weights``, ``cluster_colors``,
``save_fig`` and the construction plots below ``plot_bags``.
"""
from __future__ import annotations

import os
import random
import re
from collections.abc import Sequence
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402


# ----------------------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------------------


def perform_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: Optional[int] = None,
    verbose: Optional[bool] = False,
) -> np.ndarray:
    """
    Performs UMAP dimensionality reduction on the given embeddings.

    Args:
        embeddings: The input high-dimensional embeddings.
        n_neighbors: The size of local neighborhood used for manifold approximation.
        min_dist: The effective minimum distance between embedded points.
        n_components: The dimension of the space to embed into.
        random_state: Seed for reproducibility (disables UMAP's parallelism).
        verbose: Whether to print progress messages.

    Returns:
        The UMAP-reduced embeddings.
    """

    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        verbose=verbose,
    )
    umap_embeddings = reducer.fit_transform(embeddings)

    return umap_embeddings


def perform_kmeans(
    embeddings: np.ndarray, n_clusters: int = 5, metric: str = "euclidean", random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs KMeans clustering on the given embeddings.

    Args:
        embeddings: The input embeddings.
        n_clusters: The number of clusters to form.
        metric: The distance metric to use for computing distances.
        random_state: Seed for reproducibility.

    Returns:
        cluster_labels: The labels of each point indicating cluster membership.
        distances: The distances of each point to each cluster centroid.
        centroids: The coordinates of the cluster centroids.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    distances = cdist(embeddings, centroids, metric=metric)

    return cluster_labels, distances, centroids


def find_closest_cells_to_clusters(
    cluster_labels: np.ndarray, distances: np.ndarray, num_per_cluster: int = 5000
) -> np.ndarray:
    """
    Finds the indices of the closest cells to each cluster centroid.

    Args:
        cluster_labels: The labels of each point indicating cluster membership.
        distances: The distances of each point to each cluster centroid.
        num_per_cluster: The number of closest cells to select per cluster.

    Returns:
        The indices of the closest cells, grouped by cluster and sorted by distance.
    """

    selected_indices = []
    n_clusters = len(np.unique(cluster_labels))
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_distances = distances[cluster_indices, i]
        closest_indices = cluster_indices[np.argsort(cluster_distances)[:num_per_cluster]]
        selected_indices.extend(closest_indices)

    return np.asarray(selected_indices)


# ----------------------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------------------


def sort_key(cell_id: str) -> Tuple[float, float]:
    """Sorts '123' before '123-1' before '124'."""

    match = re.match(r"(\d+)(?:-(\d+))?", cell_id)
    if match:
        main_id = int(match.group(1))
        sub_id = int(match.group(2)) if match.group(2) else -1
        return (main_id, sub_id)
    return (float("inf"), float("inf"))


def duplicate_cluster(
    cell_ids: np.ndarray, cluster_labels: np.ndarray, cluster: int, suffix: str = "-1"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adds a copy of one cluster as a new cluster: same cells, ids suffixed with ``suffix``,
    label = number of clusters (e.g. 6 when there are clusters 0-5).

    Args:
        cell_ids: Array of cell IDs.
        cluster_labels: Array of cluster labels corresponding to cell_ids.
        cluster: The cluster to duplicate.
        suffix: Suffix appended to the ids of the copies.

    Returns:
        The extended cell ids and cluster labels.
    """

    n_clusters = len(np.unique(cluster_labels))
    cluster_indices = np.where(cluster_labels == cluster)[0]
    duplicate_ids = np.char.add(cell_ids[cluster_indices].astype(str), suffix)

    new_ids = np.concatenate([cell_ids.astype(str), duplicate_ids])
    new_labels = np.concatenate([cluster_labels, np.full(len(cluster_indices), n_clusters)])

    return new_ids, new_labels


def draw_imbalanced_weights(
    n_clusters: int, seed: int = 42, cap: float = 0.60, min_cells: int = 30, total_cells: int = 3000
) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """
    Draws a random imbalanced cluster distribution shaped like the legacy 6-cluster one
    (one dominant cluster, a few small ones, the rest mid-sized).

    Roles: 1 dominant, max(1, round(K/4)) small, the rest mid. Relative weights: mid ~ U(0.5, 1.5),
    small ~ U(0.15, 0.35), dominant ~ U(6, 10). The dominant share is capped at ``cap`` (the
    excess goes to the mid clusters) and every cluster gets at least ``min_cells`` of
    ``total_cells``. Roles are assigned to random cluster indices.

    Args:
        n_clusters: Number of clusters K.
        seed: Seed; the generator is seeded with (seed, K).
        cap: Maximum share of the dominant cluster.
        min_cells: Minimum number of cells requested per cluster.
        total_cells: Cells drawn for the dataset (mean cells per spot x number of spots).

    Returns:
        weights: Array of K weights summing to 1, in cluster order.
        roles: Cluster indices per role ("dominant", "mid", "small").
    """

    rng = np.random.default_rng([seed, n_clusters])
    n_small = max(1, int(n_clusters / 4 + 0.5))
    n_mid = n_clusters - 1 - n_small

    dom = rng.uniform(6, 10)
    mid = rng.uniform(0.5, 1.5, n_mid)
    small = rng.uniform(0.15, 0.35, n_small)

    total = dom + mid.sum() + small.sum()
    d, m, s = dom / total, mid / total, small / total
    if d > cap:
        m = m + (d - cap) * m / m.sum()
        d = cap
    s = np.maximum(s, min_cells / total_cells)

    shares = np.concatenate([[d], m, s])
    shares /= shares.sum()

    perm = rng.permutation(n_clusters)
    weights = np.empty(n_clusters)
    weights[perm] = shares
    roles = {
        "dominant": [int(perm[0])],
        "mid": sorted(int(i) for i in perm[1 : 1 + n_mid]),
        "small": sorted(int(i) for i in perm[1 + n_mid :]),
    }

    return weights, roles


def create_bags(
    cell_ids: np.ndarray,
    cluster_labels: np.ndarray,
    mean_cell_per_bag: int = 10,
    var_cell_per_bag: int = 4,
    balance: Union[List[float], str] = "auto",
    total_number_of_bags: int = 1000,
    random_state: Optional[int] = None,
    not_mixed: Optional[List[int]] = None,
) -> Tuple[List[List[str]], np.ndarray]:
    """
    Creates bags of cell IDs with controlled sampling based on cluster proportions,
    with optional constraint to not mix certain clusters.

    Args:
        cell_ids: Array of cell IDs.
        cluster_labels: Array of cluster labels corresponding to cell_ids.
        mean_cell_per_bag: Mean number of cells per bag.
        var_cell_per_bag: Variance in the number of cells per bag.
        balance: List of weights representing the desired overall distribution of clusters.
               If "auto", the cluster proportions will be used. Default is "auto".
        total_number_of_bags: Total number of bags to create.
        random_state: Seed for reproducibility.
        not_mixed: List of two cluster indices that should not appear together in any bag.

    Returns:
        bags: List of lists, where each inner list contains the cell IDs for a bag.
        cells_per_cluster: Array with the total number of sampled cells per cluster.

    Raises:
        ValueError: If a cluster is asked for more cells than it holds.
    """

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    if not isinstance(balance, list):
        if balance == "auto":
            balance = np.bincount(cluster_labels)
        else:
            raise ValueError("balance must be 'auto' or a list of weights.")
    else:
        if len(balance) != len(np.unique(cluster_labels)):
            raise ValueError("balance must have the same length as the number of clusters.")
        balance = np.array(balance)

    balance = balance / balance.sum()
    print(f"Using cluster proportions: {np.round(balance, 4)}")

    unique_clusters = np.unique(cluster_labels)
    cluster_to_indices = {cluster: np.where(cluster_labels == cluster)[0] for cluster in unique_clusters}
    id_to_cluster = dict(zip(cell_ids, cluster_labels))

    total_cells = int(mean_cell_per_bag * total_number_of_bags)
    cells_per_cluster = (balance * total_cells).astype(int)

    sampled_cells_by_cluster = {}
    for cluster, count in zip(unique_clusters, cells_per_cluster):
        indices = cluster_to_indices[cluster]
        if count > len(indices):
            raise ValueError(
                f"Cluster {cluster} is asked for {count} cells but holds only {len(indices)}; "
                "sampling with replacement would put the same cell in several spots."
            )
        sampled_indices = np.random.choice(indices, size=count, replace=False)
        sampled_cells_by_cluster[cluster] = list(cell_ids[sampled_indices])

    bags = []

    for _ in range(total_number_of_bags):
        n_cells = int(np.random.normal(mean_cell_per_bag, np.sqrt(var_cell_per_bag)))
        n_cells = max(1, n_cells)

        bag = []
        available_clusters = list(unique_clusters)
        random.shuffle(available_clusters)

        while len(bag) < n_cells and available_clusters:
            cluster = random.choice(available_clusters)

            # Check not_mixed constraint
            if not_mixed is not None:
                clusters_in_bag = {id_to_cluster[cid] for cid in bag}
                if (not_mixed[0] in clusters_in_bag and cluster == not_mixed[1]) or (
                    not_mixed[1] in clusters_in_bag and cluster == not_mixed[0]
                ):
                    available_clusters.remove(cluster)
                    continue

            if sampled_cells_by_cluster[cluster]:
                bag.append(sampled_cells_by_cluster[cluster].pop())

            # Remove the cluster from future sampling if it's empty
            if not sampled_cells_by_cluster[cluster]:
                available_clusters.remove(cluster)

        bags.append(bag)

    random.shuffle(bags)

    return bags, cells_per_cluster


def get_bag_proportions(bags: List[List[str]], cell_ids: np.ndarray, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Calculates the cluster proportions for each bag.

    Args:
        bags: List of lists, where each inner list contains the cell IDs for a bag.
        cell_ids: Array of cell IDs.
        cluster_labels: Array of cluster labels corresponding to cell_ids.

    Returns:
        bag_prop_df: DataFrame with shape (n_bags, n_clusters) containing cluster proportions per bag.
    """

    id_to_cluster = dict(zip(cell_ids, cluster_labels))
    columns = sorted(set(id_to_cluster.values()))

    bag_prop_df = pd.DataFrame(0.0, index=range(len(bags)), columns=columns)

    for bag_id, bag in enumerate(bags):
        clusters_in_bag = [id_to_cluster[cell_id] for cell_id in bag]
        unique_clusters, counts = np.unique(clusters_in_bag, return_counts=True)
        proportions = counts / len(bag)
        for cluster_id, proportion in zip(unique_clusters, proportions):
            bag_prop_df.at[bag_id, cluster_id] = proportion

    bag_prop_df.columns = [f"Cluster {cluster_id}" for cluster_id in columns]
    bag_prop_df.index = bag_prop_df.index.astype(str)

    return bag_prop_df


def add_perturbation(proportions_df: pd.DataFrame, strength: float, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Adds perturbation to a proportion dataframe.

    Args:
        proportions_df: pd.DataFrame with shape (n_spots, n_cell_types), each row sums to 1
        strength: float in [0, 1], 0 = no perturbation, 1 = full random
        random_state: Seed of the noise.

    Returns:
        perturbed_df: pd.DataFrame with same shape, rows still sum to 1
    """

    if not (0 <= strength <= 1):
        raise ValueError("strength must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    noise = rng.random(proportions_df.shape)
    mixed = (1 - strength) * proportions_df.values + strength * noise
    mixed /= mixed.sum(axis=1, keepdims=True)

    perturbed_df = pd.DataFrame(mixed, index=proportions_df.index, columns=proportions_df.columns)

    return perturbed_df


# ----------------------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------------------


def save_fig(fig: plt.Figure, stem: str, dpi: int = 200) -> None:
    """Saves ``fig`` as ``stem.png`` and ``stem.svg`` and closes it."""

    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def cluster_colors(n_clusters: int) -> List[tuple]:
    """Fixed colour per cluster index: tab10 up to 10 clusters, else the 10 dark tab20 hues then the 10 light ones."""

    if n_clusters <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n_clusters)]
    cmap = plt.get_cmap("tab20")
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    return [cmap(order[i]) for i in range(n_clusters)]


def _cluster_legend(ax: plt.Axes, colors: Sequence[tuple], labels: Sequence[str], **kwargs) -> None:
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=c) for c in colors]
    ax.legend(handles, labels, frameon=False, **kwargs)


def plot_cells_per_cluster(
    image_dict: dict,
    cell_ids: np.ndarray,
    cluster_labels: np.ndarray,
    selection: str = "top",
    nrows: int = 1,
    ncols: int = 1,
    random_state: Optional[int] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plots 8x8 mosaics per cluster in a grid of cluster mosaics (nrows x ncols).

    Args:
        image_dict: Dictionary mapping cell IDs to images (as uint8 tensors, CxHxW).
        cell_ids: Array of cell IDs.
        cluster_labels: Array of cluster labels corresponding to cell_ids.
        selection: 'top', 'random', or 'bottom' to select which cells to display per cluster
                   ('top' = first 64 in the given order, i.e. closest to the centroid).
        nrows: Number of rows in the grid of cluster mosaics.
        ncols: Number of columns in the grid of cluster mosaics.
        random_state: Seed for selection='random'.
        title: Optional figure title.

    Returns:
        The matplotlib figure.
    """

    cluster_order = np.unique(cluster_labels)
    n_clusters = len(cluster_order)
    if n_clusters > nrows * ncols:
        raise ValueError(f"Number of clusters ({n_clusters}) exceeds layout capacity ({nrows}x{ncols})")

    rng = np.random.default_rng(random_state)
    gap = 2  # white pixels between cells

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4.3), squeeze=False, layout="constrained"
    )
    for ax in axes.flat:
        ax.axis("off")

    for cluster_idx, cluster in enumerate(cluster_order):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        n_show = min(len(cluster_indices), 64)

        if selection == "top":
            selected_indices = cluster_indices[:n_show]
        elif selection == "random":
            selected_indices = np.sort(rng.choice(cluster_indices, size=n_show, replace=False))
        elif selection == "bottom":
            selected_indices = cluster_indices[-n_show:]
        else:
            raise ValueError("selection must be 'top', 'random', or 'bottom'.")

        # one 8x8 mosaic image per cluster
        images = [image_dict[cell_id].numpy().transpose(1, 2, 0) for cell_id in cell_ids[selected_indices]]
        h, w = images[0].shape[:2]
        mosaic = np.full((8 * h + 7 * gap, 8 * w + 7 * gap, 3), 255, dtype=np.uint8)
        for i, img in enumerate(images):
            r, c = divmod(i, 8)
            mosaic[r * (h + gap) : r * (h + gap) + h, c * (w + gap) : c * (w + gap) + w] = img

        ax = axes[cluster_idx // ncols, cluster_idx % ncols]
        ax.imshow(mosaic, interpolation="nearest")
        ax.set_title(f"Cluster {cluster}", loc="left", fontsize=13)

    if title:
        fig.suptitle(title, fontsize=15)

    return fig


def plot_cluster_distribution(
    target_counts: np.ndarray,
    actual_counts: np.ndarray,
    cluster_names: Sequence[str],
    title: str = "",
) -> plt.Figure:
    """
    Bar chart of cells per cluster: cells drawn for the dataset (target, from the weights)
    next to cells that ended up in spots (actual).

    Args:
        target_counts: Cells drawn per cluster.
        actual_counts: Cells present in spots per cluster.
        cluster_names: Cluster names, in the same order.
        title: Title for the plot.

    Returns:
        The matplotlib figure.
    """

    n = len(cluster_names)
    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * n + 2), 4.5))
    ax.bar(
        x - width / 2,
        target_counts,
        width * 0.95,
        color="#b8b8b8",
        label=f"target (drawn, n={int(np.sum(target_counts))})",
    )
    ax.bar(
        x + width / 2,
        actual_counts,
        width * 0.95,
        color="#3a6ea5",
        label=f"actual (in spots, n={int(np.sum(actual_counts))})",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("Cluster ", "") for c in cluster_names])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of cells")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)

    return fig


def plot_spot_sizes(sizes: np.ndarray, mean: float, var: float, title: str = "") -> plt.Figure:
    """Histogram of the number of cells per spot, with the realised mean."""

    sizes = np.asarray(sizes)
    x_range = np.arange(0, sizes.max() + 1)
    counts = np.array([(sizes == v).sum() for v in x_range])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x_range, counts, width=0.85, color="#3a6ea5")
    ax.axvline(sizes.mean(), color="#222222", linestyle="--", linewidth=1.5)
    ax.set_ylim(0, counts.max() * 1.3)
    ax.text(
        0.02,
        0.97,
        f"target: N(mean={mean}, var={var})\nrealised: mean={sizes.mean():.2f}, var={sizes.var(ddof=1):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", pad=2),
        zorder=10,
    )
    ax.set_xlabel("Cells per spot")
    ax.set_ylabel("Number of spots")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)

    return fig


def plot_bags(
    bags: list,
    cell_ids: list,
    cluster_labels: list,
    title: str = "",
    fig_edge_size: int = 20,
    random_state: Optional[int] = None,
) -> plt.Figure:
    """
    Visualizes spatial 'bags' of cells, where each bag contains cells colored by cluster.

    Args:
        bags: Each sublist contains cell IDs belonging to a bag.
        cell_ids: List or array of cell IDs corresponding to cluster_labels.
        cluster_labels: Cluster assignment for each cell ID.
        title: Title for the figure.
        fig_edge_size: Size of the figure edge (controls overall scale).
        random_state: Seed for the cell positions inside each bag.

    Returns:
        The matplotlib figure.
    """

    id_to_cluster = dict(zip(cell_ids, cluster_labels))
    unique_clusters = np.unique(cluster_labels)
    colors = cluster_colors(len(unique_clusters))
    rng = np.random.default_rng(random_state)

    num_bags = len(bags)
    grid_size = int(np.ceil(np.sqrt(num_bags)))

    bag_radius = 0.4 * (fig_edge_size / 20) / float(grid_size ** (1 / 3)) * 3
    bag_radius = min(bag_radius, 0.47)
    cell_size = 20 * (fig_edge_size / 20) / grid_size * 32

    fig, ax = plt.subplots(figsize=(fig_edge_size, fig_edge_size))
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=22, fontweight="bold")

    color_of = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    xs, ys, cs = [], [], []
    for bag_id, bag in enumerate(bags):
        row = bag_id // grid_size
        col = bag_id % grid_size
        x = col
        y = grid_size - 1 - row

        circle = Circle((x, y), radius=bag_radius, edgecolor="black", facecolor="none", lw=1.5)
        ax.add_patch(circle)

        # random position within the bag
        for cell_id in bag:
            angle = 2 * np.pi * rng.random()
            radius = bag_radius * np.sqrt(rng.random())
            xs.append(x + radius * np.cos(angle))
            ys.append(y + radius * np.sin(angle))
            cs.append(color_of[id_to_cluster[cell_id]])

    ax.scatter(xs, ys, color=cs, s=cell_size, edgecolor="black", linewidth=0.4)

    _cluster_legend(
        ax,
        colors,
        [f"Cluster {c}" for c in unique_clusters],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=14,
    )
    for side in ax.spines.values():
        side.set_visible(False)

    return fig


def plot_cooccurrence(
    bags: List[List[str]], id_to_cluster: dict, cluster_names: Sequence[str], title: str = ""
) -> plt.Figure:
    """
    Heatmap of the number of spots containing both cluster i and cluster j
    (diagonal: spots containing cluster i).
    """

    n = len(cluster_names)
    presence = np.zeros((len(bags), n), dtype=int)
    for b, bag in enumerate(bags):
        for cell_id in bag:
            presence[b, id_to_cluster[cell_id]] = 1
    cooc = presence.T @ presence

    short = [c.replace("Cluster ", "") for c in cluster_names]
    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 2.2))
    sns.heatmap(
        pd.DataFrame(cooc, index=short, columns=short),
        cmap="Blues",
        annot=n <= 10,
        fmt="d",
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spots"},
        ax=ax,
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    ax.set_title(title)

    return fig


def plot_perturbation(base: pd.DataFrame, perturbed: Dict[float, pd.DataFrame], title: str = "") -> plt.Figure:
    """Original vs perturbed proportions (every spot x cluster entry), one panel per strength."""

    n = len(perturbed)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.9), sharex=True, sharey=True, squeeze=False)
    for ax, (strength, df) in zip(axes[0], perturbed.items()):
        ax.plot([0, 1], [0, 1], color="#999999", linewidth=1, zorder=1)
        ax.scatter(
            base.values.ravel(),
            df.values.ravel(),
            s=6,
            color="#3a6ea5",
            alpha=0.35,
            linewidth=0,
            rasterized=True,
            zorder=2,
        )
        mae = np.abs(df.values - base.values).mean()
        ax.set_title(f"strength {strength}\nmean |diff| = {mae:.3f}", fontsize=11)
        ax.set_xlabel("Original proportion")
        ax.set_aspect("equal")
    axes[0, 0].set_ylabel("Perturbed proportion")
    fig.suptitle(title, y=1.04)

    return fig


def plot_umap(
    umap_xy: np.ndarray, cluster_labels: np.ndarray, pool_mask: np.ndarray, title: str = "", random_state: int = 0
) -> plt.Figure:
    """Two UMAP panels: every cell coloured by K-means cluster, and only the retained cells."""

    n_clusters = int(cluster_labels.max()) + 1
    colors = np.array(cluster_colors(n_clusters))
    order = np.random.default_rng(random_state).permutation(len(umap_xy))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharex=True, sharey=True)
    ax = axes[0]
    ax.scatter(
        umap_xy[order, 0], umap_xy[order, 1], c=colors[cluster_labels[order]], s=0.2, linewidth=0, rasterized=True
    )
    ax.set_title(f"All cells (n={len(umap_xy):,})")

    ax = axes[1]
    ax.scatter(umap_xy[:, 0], umap_xy[:, 1], color="#dddddd", s=0.2, linewidth=0, rasterized=True)
    sel = order[pool_mask[order]]
    ax.scatter(umap_xy[sel, 0], umap_xy[sel, 1], c=colors[cluster_labels[sel]], s=0.6, linewidth=0, rasterized=True)
    ax.set_title(f"Retained cells, closest to each centroid (n={pool_mask.sum():,})")

    for ax in axes:
        ax.set_xlabel("UMAP 1")
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax)
    axes[0].set_ylabel("UMAP 2")
    _cluster_legend(
        axes[1],
        colors,
        [f"Cluster {k}" for k in range(n_clusters)],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        ncol=1 if n_clusters <= 10 else 2,
    )
    fig.suptitle(title)

    return fig


def plot_slide_map(
    background: np.ndarray, xy: np.ndarray, cluster_labels: np.ndarray, n_clusters: int, title: str = ""
) -> plt.Figure:
    """Retained cells on the tissue image (xy in background pixel coordinates)."""

    colors = np.array(cluster_colors(n_clusters))
    h, w = background.shape[:2]

    fig, ax = plt.subplots(figsize=(8 * w / h + 2.5, 8))
    ax.imshow(background, alpha=0.55)
    ax.scatter(xy[:, 0], xy[:, 1], c=colors[cluster_labels], s=1.2, linewidth=0, rasterized=True)
    ax.set_axis_off()
    ax.set_title(title)
    _cluster_legend(
        ax,
        colors,
        [f"Cluster {k}" for k in range(n_clusters)],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        ncol=1 if n_clusters <= 10 else 2,
    )

    return fig


def plot_cluster_sizes(cluster_sizes: np.ndarray, pool_sizes: np.ndarray, title: str = "") -> plt.Figure:
    """K-means cluster sizes (all cells) and the number of retained cells per cluster."""

    n = len(cluster_sizes)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * n + 2), 4.5))
    ax.bar(x, cluster_sizes, width=0.8, color="#b8b8b8", label="all cells in the K-means cluster")
    ax.bar(x, pool_sizes, width=0.8, color="#3a6ea5", label="retained (closest to centroid)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in range(n)])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of cells")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)

    return fig
