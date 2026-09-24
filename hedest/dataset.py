from __future__ import annotations

import torch
from torch.utils.data import Dataset


class EmbedDataset(Dataset):
    """Dataset for loading individual cell embeddings from a pre-saved embed_dict.pt"""

    def __init__(self, embed_dict: dict[str, torch.Tensor]) -> None:
        self.embed_dict = embed_dict
        self.cell_ids = list(embed_dict.keys())

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        embedding = self.embed_dict[cell_id]

        return embedding, cell_id


class SpotEmbedDataset(Dataset):
    """Dataset for loading spots with their corresponding cell embeddings and proportions."""

    def __init__(self, spot_dict, spot_prop_df, embed_dict):
        self.spot_dict = spot_dict
        self.spot_prop_df = spot_prop_df
        self.embed_dict = embed_dict
        self.spot_ids = list(spot_dict.keys())

        self.embed_size = next(iter(embed_dict.values())).numel()

    def __len__(self):
        return len(self.spot_ids)

    def __getitem__(self, idx):
        spot_id = self.spot_ids[idx]
        cell_ids = self.spot_dict[spot_id]
        embeddings = torch.stack([self.embed_dict[cell_id] for cell_id in cell_ids])
        proportions = torch.tensor(self.spot_prop_df.loc[spot_id].values, dtype=torch.float32)

        # No bag index here: a spot is identified by its position in the batch, which only
        # `custom_collate` knows. See hedest.dataset_utils.custom_collate.
        return {"embeddings": embeddings, "proportions": proportions}


class CellProbDataset(Dataset):
    """Dataset for cell probabilities used during PPSA."""

    def __init__(self, p_cell: torch.Tensor, p_local: torch.Tensor, beta: torch.Tensor):
        self.p_cell = p_cell  # (N, n_types)
        self.p_local = p_local  # (N, n_types)
        self.beta = beta  # (N,)

    def __len__(self):
        return self.p_cell.size(0)

    def __getitem__(self, idx):
        return self.p_cell[idx], self.p_local[idx], self.beta[idx], idx
