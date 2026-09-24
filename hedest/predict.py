from __future__ import annotations

from typing import Dict
from typing import List

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from hedest.config import TqdmToLogger
from hedest.dataset import EmbedDataset
from hedest.model.cell_classifier import CellClassifier

tqdm_out = TqdmToLogger(logger, level="INFO")


def predict_slide(
    model: CellClassifier,
    embed_dict: Dict[str, torch.Tensor],
    ct_list: List[str],
    batch_size: int = 1024,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Predicts the cell type probabilities for all cells in a slide.

    Args:
        model: The trained model to use for predictions.
        embed_dict: A dictionary where keys are cell IDs and values are cell embeddings.
        ct_list: List of cell type names.
        batch_size: Batch size for prediction.
        verbose: Whether to display progress.

    Returns:
        A DataFrame where rows correspond to cell IDs and columns to cell type probabilities.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Device used: {device}")

    model.eval()
    model = model.to(device)
    cell_prob = []

    dataset = EmbedDataset(embed_dict)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for embeddings, cell_ids in tqdm(
            dataloader, desc="Predicting on cells", unit="batch", file=tqdm_out, disable=(not verbose)
        ):
            embeddings = embeddings.to(device)
            outputs = model(embeddings)

            for cell_id, prob_vector in zip(cell_ids, outputs):
                cell_prob.append(
                    {
                        "cell_id": cell_id,
                        **{ct_list[i]: prob for i, prob in enumerate(prob_vector.cpu().tolist())},
                    }
                )

    cell_prob_df = pd.DataFrame(cell_prob)
    cell_prob_df.set_index("cell_id", inplace=True)

    return cell_prob_df
