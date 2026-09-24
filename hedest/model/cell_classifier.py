from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from hedest.model.base_cell_classifier import BaseCellClassifier


class CellClassifier(BaseCellClassifier):
    def __init__(
        self,
        num_classes: int,
        embed_size: int,
        hidden_dims: list = [512, 256],
        norm: bool = False,
        dropout: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Cell classifier applied to cell embeddings.

        The embeddings come from a feature extractor run beforehand (H-Optimus-0, MoCo,
        ...), so the classifier itself is a multilayer perceptron ending on a softmax
        over the cell types.

        Args:
            num_classes (int): Number of output classes.
            embed_size (int): Size of the input embeddings.
            hidden_dims (list): List of hidden dimensions for the fully connected layers.
            norm (bool): Whether to add a LayerNorm layer.
            dropout (float): Dropout rate.
            device (torch.device): Device to run the model on.
        """

        super().__init__(num_classes, device)
        self.embed_size = embed_size
        self.hidden_dims = hidden_dims
        self.norm = norm
        self.dropout = dropout

        if self.embed_size is None:
            raise ValueError("embed_size must be provided.")

        self.backbone = nn.Sequential()
        input_dim = self.embed_size
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.backbone.add_module(f"fc_{i}", nn.Linear(input_dim, hidden_dim))
            if self.norm:
                self.backbone.add_module(f"layernorm_{i}", nn.LayerNorm(hidden_dim))
            self.backbone.add_module(f"relu_{i}", nn.ReLU())
            if self.dropout > 0.0:
                self.backbone.add_module(f"dropout_{i}", nn.Dropout(self.dropout))
            input_dim = hidden_dim

        self.backbone.add_module("final", nn.Linear(input_dim, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)

        return F.softmax(features, dim=1)
