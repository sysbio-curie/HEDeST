from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from celltype_permutation import hierarchical_permutations
from utils import aggregate_stats
from utils import compute_stats
from utils import generate_X_perm


def prepare_data(ground_truth: pd.DataFrame, spot_dict: dict, embeddings: dict):
    """
    Prepares data matrices A, B, X_perm, X for hierarchical permutation.

    Args:
        ground_truth: DataFrame with one-hot encoded cell type labels.
        spot_dict: Dictionary mapping spot IDs to lists of cell IDs.
        embeddings: Dictionary mapping cell IDs to embedding tensors.

    Returns:
        A: Spot-cell assignment matrix (M x N).
        B: Cell feature matrix (N x K).
        X_perm: Permutable cell type label matrix (M x N).
        X: Flat cell type label vector (N,).
    """

    # 1. Unique sorted cell IDs across both spot_dict and ground_truth
    cell_ids = sorted(set(ground_truth.index.astype(str)) & set(sum(spot_dict.values(), [])))
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    N = len(cell_ids)

    # 2. Sorted spot IDs
    spot_ids = sorted(spot_dict.keys())
    spot_id_to_idx = {sid: i for i, sid in enumerate(spot_ids)}
    M = len(spot_ids)

    # 3. Map one-hot labels to class indices
    # cell_type_names = ground_truth.columns.tolist()
    # L = len(cell_type_names)
    gt_labels = ground_truth.loc[ground_truth.index.astype(str).isin(cell_ids)]
    label_array = gt_labels.to_numpy()
    type_indices = label_array.argmax(axis=1) + 1  # get integer labels: 1, 2, ..., L

    # Map cell_id → class index
    cell_id_to_label = {str(cid): type_idx for cid, type_idx in zip(gt_labels.index.astype(str), type_indices)}

    # 4. B: Features per cell (N x K)
    K = len(next(iter(embeddings.values())))  # 2048
    B = np.zeros((N, K))

    for i, cid in enumerate(cell_ids):
        if cid in embeddings:
            B[i] = embeddings[cid].detach().cpu().numpy()
        else:
            raise ValueError(f"Cell ID {cid} found in ground_truth or spot_dict but not in embeddings.")

    # 5. A and X_sparse
    A = np.zeros((M, N), dtype=int)
    X_sparse = np.zeros((M, N), dtype=int)

    for sid, cell_list in spot_dict.items():
        m = spot_id_to_idx[sid]
        for cid in cell_list:
            if cid in cell_id_to_idx and cid in cell_id_to_label:
                n = cell_id_to_idx[cid]
                A[m, n] = 1
                X_sparse[m, n] = cell_id_to_label[cid]

    # 6. X: flat label vector (N,)
    X = np.zeros(N, dtype=int)
    for cid, n in cell_id_to_idx.items():
        if cid in cell_id_to_label:
            X[n] = cell_id_to_label[cid]

    # 7. New organization of matrices
    new_order = []
    seen = set()

    for row in X_sparse:
        cell_indices = np.where(row > 0)[0]
        for idx in cell_indices:
            if idx not in seen:
                new_order.append(idx)
                seen.add(idx)

    X_sparse = X_sparse[:, new_order]
    A = A[:, new_order]
    B = B[new_order]
    X = X[new_order]

    X_perm, _ = generate_X_perm(X_sparse.copy())

    return A, B, X_perm, X


def main(
    data_path: str,
    gt_filename: str,
    spot_dict_filename: str,
    embeddings_filename: str,
    n_iter: int,
    output_xlsx: str,
    seed: int | None = None,
):
    """
    Main function to run hierarchical permutation evaluation.

    Args:
        data_path: Base directory containing data files.
        gt_filename: Filename for ground truth CSV.
        spot_dict_filename: Filename for spot dictionary JSON.
        embeddings_filename: Filename for embeddings PT file.
        n_iter: Number of repetitions.
        output_xlsx: Path to output Excel file.
        seed: Seed of the initial random permutation, so that a repetition can be reproduced
              and run on its own. Left unseeded when None, as before.
    """

    if seed is not None:
        np.random.seed(seed)
        print(f"-> numpy seeded with {seed}")

    ground_truth_path = os.path.join(data_path, gt_filename)
    spot_dict_path = os.path.join(data_path, spot_dict_filename)
    embeddings_path = os.path.join(data_path, embeddings_filename)
    print(f"-> Loading data from {data_path}")

    ground_truth = pd.read_csv(ground_truth_path, index_col=0)
    embeddings = torch.load(embeddings_path)

    with open(spot_dict_path, "r") as file:
        spot_dict = json.load(file)
    print("-> Data loaded successfully")

    metrics_before_list = []
    metrics_after_list = []

    for i in range(n_iter):
        print(f"-> Iteration {i + 1}/{n_iter}")
        A, B, X_perm, X = prepare_data(ground_truth, spot_dict, embeddings)
        X_global = hierarchical_permutations(A, X_perm, B)
        metrics_before = compute_stats(X, X_perm, per_class=False)
        metrics_after = compute_stats(X, X_global, per_class=False)
        metrics_before_list.append(metrics_before)
        metrics_after_list.append(metrics_after)

    mean_values_before, ci_values_before = aggregate_stats(metrics_before_list)
    mean_values_after, ci_values_after = aggregate_stats(metrics_after_list)

    df_summary = pd.DataFrame(
        [
            {"state": "before", **mean_values_before, **ci_values_before},
            {"state": "after", **mean_values_after, **ci_values_after},
        ]
    )

    df_before_all = pd.DataFrame(metrics_before_list)
    df_after_all = pd.DataFrame(metrics_after_list)

    with pd.ExcelWriter(output_xlsx) as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_before_all.to_excel(writer, sheet_name="Metrics Before", index=False)
        df_after_all.to_excel(writer, sheet_name="Metrics After", index=False)

    print(f"-> Results saved to {output_xlsx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hierarchical permutation evaluation on spatial transcriptomics data."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Base directory containing data.")
    parser.add_argument("--gt_filename", type=str, required=True, help="CSV file containing ground truth labels.")
    parser.add_argument("--spot_dict_filename", type=str, required=True, help="JSON file with spot dictionary.")
    parser.add_argument("--embeddings_filename", type=str, required=True, help="PT file with embeddings.")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of repetitions")
    parser.add_argument("--output_xlsx", type=str, default="results.csv", help="Path to output CSV")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the initial random permutation.")

    args = parser.parse_args()
    main(
        args.data_path,
        args.gt_filename,
        args.spot_dict_filename,
        args.embeddings_filename,
        args.n_iter,
        args.output_xlsx,
        args.seed,
    )
