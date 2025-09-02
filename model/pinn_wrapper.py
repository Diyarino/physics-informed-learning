# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:37:30 2025

@author: Altinses
"""

# %% imports

import torch

from utils.physics_pde_operators import PDEOperator
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# %% pinn wrapper

@dataclass
class LossWeights:
    """Weights for different PINN loss components."""

    physics: float = 1.0
    boundary: float = 1.0
    supervised: float = 0.0


class PINN(torch.nn.Module):
    """Physics-Informed Neural Network wrapper.

    This class encapsulates a neural network and a PDE operator, exposing
    convenient methods to compute composite losses.

    Args:
        model: The neural network architecture (e.g., an instance of `MLP`).
        pde: PDE operator implementing `residual(x, model)`.
        loss_weights: Weights for physics/boundary/supervised losses.
    """

    def __init__(self, model: torch.nn.Module, pde: PDEOperator, loss_weights: LossWeights) -> None:
        super().__init__()
        self.model = model
        self.pde = pde
        self.loss_weights = loss_weights
        self.mse = torch.nn.MSELoss()

    def physics_loss(self, x_collocation: torch.Tensor) -> torch.Tensor:
        """Compute MSE of the PDE residual at collocation points.

        Args:
            x_collocation: Collocation points (N, D) with `requires_grad=True`.
        Returns:
            Scalar tensor representing the physics loss.
        """
        r = self.pde.residual(x_collocation, self.model)
        return torch.mean(r**2)

    def boundary_loss(self, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        """Compute MSE of boundary condition residuals.

        Args:
            xb: Boundary coordinates (M, D).
            yb: Target boundary values (M, 1).
        Returns:
            Scalar tensor representing boundary loss.
        """
        preds = self.model(xb)
        return self.mse(preds, yb)

    def supervised_loss(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Compute MSE for supervised anchor points (optional).

        Args:
            xs: Supervised coordinates (K, D).
            ys: Target values (K, 1).
        Returns:
            Scalar tensor representing supervised loss.
        """
        preds = self.model(xs)
        return self.mse(preds, ys)

    def total_loss(
        self,
        x_collocation: torch.Tensor,
        boundary_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        supervised_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute a dictionary of individual and weighted losses.

        Args:
            x_collocation: Collocation points for physics residual.
            boundary_batch: Optional tuple (xb, yb) for boundary loss.
            supervised_batch: Optional tuple (xs, ys) for supervised loss.
        Returns:
            Dictionary with keys: 'physics', 'boundary', 'supervised', 'total'.
        """
        losses: Dict[str, torch.Tensor] = {}
        losses["physics"] = self.physics_loss(x_collocation)

        if boundary_batch is not None:
            xb, yb = boundary_batch
            losses["boundary"] = self.boundary_loss(xb, yb)
        else:
            losses["boundary"] = torch.tensor(0.0, device=x_collocation.device)

        if supervised_batch is not None:
            xs, ys = supervised_batch
            losses["supervised"] = self.supervised_loss(xs, ys)
        else:
            losses["supervised"] = torch.tensor(0.0, device=x_collocation.device)

        total = (
            self.loss_weights.physics * losses["physics"]
            + self.loss_weights.boundary * losses["boundary"]
            + self.loss_weights.supervised * losses["supervised"]
        )
        losses["total"] = total
        return losses