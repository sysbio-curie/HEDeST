from __future__ import annotations

from typing import Union

import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def split_data(
    spot_dict: dict[str, list[str]],
    spot_prop_df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    rs: int = 42,
) -> tuple[dict[str, list[str]], pd.DataFrame, dict[str, list[str]], pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    """
    Splits data into training, validation, and testing sets.

    Args:
        spot_dict: Dictionary containing {spot_id: list of cell IDs}.
        spot_prop_df: DataFrame where each row corresponds to a spot and columns
                      represent cell type proportions.
        train_size: Proportion of the dataset to use for training. Defaults to 0.7.
        val_size: Proportion of the dataset to use for validation. Defaults to 0.15.
        rs: Random state for reproducibility. Defaults to 42.

    Returns:
        A tuple containing:
            - Training set dictionary and corresponding proportions.
            - Validation set dictionary and corresponding proportions.
            - Testing set dictionary and corresponding proportions.
    """

    assert train_size + val_size <= 1, "Train size + validation size must not exceed 1."

    spot_ids = list(spot_dict.keys())

    train_ids, temp_ids = train_test_split(spot_ids, train_size=train_size, random_state=rs)
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_size / (1 - train_size), random_state=rs)

    train_spot_dict = {spot: spot_dict[spot] for spot in train_ids}
    val_spot_dict = {spot: spot_dict[spot] for spot in val_ids}
    test_spot_dict = {spot: spot_dict[spot] for spot in test_ids}

    train_proportions = spot_prop_df.loc[train_ids]
    val_proportions = spot_prop_df.loc[val_ids]
    test_proportions = spot_prop_df.loc[test_ids]

    return train_spot_dict, train_proportions, val_spot_dict, val_proportions, test_spot_dict, test_proportions


def pp_prop(spot_prop: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Preprocesses spot proportions by normalizing each row to sum to 1.

    Args:
        spot_prop: A DataFrame where each row corresponds to a spot and columns represent cell type proportions.
                   If a string is provided, it is treated as a file path, and the DataFrame is read from the file.

    Returns:
        A normalized DataFrame where each row sums to 1.
    """

    if isinstance(spot_prop, str):
        spot_prop = pd.read_csv(spot_prop, index_col=0)

    spot_prop.index = spot_prop.index.astype(str)
    row_sums = spot_prop.sum(axis=1)
    spot_prop = spot_prop.div(row_sums, axis=0)

    return spot_prop


def custom_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom collate function to combine a list of spots into a batch.

    The cells of a spot are concatenated as one contiguous block, in the order the spots
    appear in the batch, so spot number ``b`` of the batch owns block number ``b``. The
    bag indices are therefore built here, positionally: ``bag_indices[i] == b`` means that
    cell ``i`` belongs to the spot whose proportions are ``proportions[b]``. This is what
    makes ``scatter_mean(outputs, bag_indices)`` line up row by row with ``proportions``.

    Args:
        batch: A list of dictionaries, each containing 'embeddings' and 'proportions'.
    """

    counts = torch.tensor([b["embeddings"].shape[0] for b in batch], dtype=torch.long)

    embeddings = torch.cat([b["embeddings"] for b in batch])
    proportions = torch.stack([b["proportions"] for b in batch])
    bag_indices = torch.repeat_interleave(torch.arange(len(batch), dtype=torch.long), counts)

    return {"embeddings": embeddings, "proportions": proportions, "bag_indices": bag_indices}
