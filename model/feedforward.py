# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:10:42 2025

@author: Altinses
"""

# %% imports

import torch
from typing import Callable, Iterable

# %% model

class FeedForwardNN(torch.nn.Module):
    """A flexible fully-connected network for PINNs.

    The network maps input coordinates (e.g., x) to the solution field u(x).

    Args:
        in_dim: Input dimensionality (e.g., 1 for x).
        out_dim: Output dimensionality (e.g., 1 for scalar u).
        hidden_layers: Number of hidden layers.
        hidden_units: Units per hidden layer.
        activation: Activation function class (default: nn.Tanh).
        use_weight_norm: Whether to apply weight normalization to linear layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_layers: int = 4,
        hidden_units: int = 64,
        activation: Callable[[], torch.nn.Module] = torch.nn.Tanh,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()

        def linear(in_f: int, out_f: int) -> torch.nn.Module:
            layer = torch.nn.Linear(in_f, out_f)
            if use_weight_norm:
                return torch.nn.utils.weight_norm(layer)
            return layer

        layers: Iterable[torch.nn.Module] = []
        layers = [linear(in_dim, hidden_units), activation()]
        for _ in range(hidden_layers - 1):
            layers += [linear(hidden_units, hidden_units), activation()]
        layers += [linear(hidden_units, out_dim)]
        self.net = torch.nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier/Glorot for stability in PINNs."""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input coordinates with shape (N, in_dim).
        Returns:
            Predicted field with shape (N, out_dim).
        """
        return self.net(x)
    
# %% test

if __name__ == '__main__':

    model = FeedForwardNN(in_dim = 10)
    outs = model(torch.rand(16,10))



    
    
    