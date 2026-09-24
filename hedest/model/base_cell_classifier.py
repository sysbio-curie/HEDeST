from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
from torch_scatter import scatter_mean

from hedest.loss import kl_divergence
from hedest.loss import l1_loss
from hedest.loss import l2_loss


class BaseCellClassifier(nn.Module, ABC):
    def __init__(
        self,
        num_classes: int,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Base class for image classifiers.

        Args:
            num_classes: Number of output classes.
        """

        super().__init__()
        self.num_classes = num_classes
        self.device = device

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def compute_loss(
        self,
        outputs: Tensor,
        bag_indices: Tensor,
        true_proportions: Tensor,
        divergence: str = "l1",
        alpha: float = 0.5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the combined loss of the model, which includes a divergence loss
        and an optional maximum probability loss.

        Args:
            outputs: The output probabilities from the model (num_cells x num_classes).
            bag_indices: Indices indicating which bag each cell belongs to (num_cells).
                         They must be row indices into `true_proportions`, i.e. values in
                         [0, num_bags), so that averaging the cells of bag b yields the
                         prediction compared against `true_proportions[b]`. They are built
                         by `hedest.dataset_utils.custom_collate`.
            true_proportions: The ground-truth class proportions (num_bags x num_classes).
            divergence: Type of divergence loss to use. Options are "l1", "l2", or "kl".
            alpha: Weight for the max probability loss. Should be in the range [0, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Total loss combining divergence and max probability loss.
                - Maximum probability loss.
                - Divergence loss.
        """

        max_probs, _ = outputs.max(dim=1)
        max_prob_loss = -torch.mean(torch.log(max_probs))

        pred_proportions = scatter_mean(outputs, bag_indices, dim=0)

        if divergence == "l1":
            divergence_loss = l1_loss(pred_proportions, true_proportions)
        elif divergence == "l2":
            divergence_loss = l2_loss(pred_proportions, true_proportions)
        elif divergence == "kl":
            divergence_loss = kl_divergence(pred_proportions, true_proportions)
        else:
            raise ValueError(f"Invalid divergence type: {divergence}. Use 'l1', 'l2', or 'kl'.")

        loss = alpha * max_prob_loss + (1 - alpha) * divergence_loss
        return loss, max_prob_loss, divergence_loss
