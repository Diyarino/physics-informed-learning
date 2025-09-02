# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:18:16 2025

@author: Altinses
"""

# %% imports

import math
import torch

from typing import Callable, Optional
from utils.diff_operators import grad_n

# %% physics pde operators

class PDEOperator:
    """Interface for computing a physics residual.

    Subclasses should implement `residual(x, model)` that returns the PDE residual
    evaluated at collocation points x.
    """

    def residual(self, x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:  # pragma: no cover - interface
        """Compute PDE residual at points `x`.

        Args:
            x: Collocation points (N, D) with `requires_grad=True`.
            model: Neural network mapping x -> u(x).
        Returns:
            Residual tensor r(x) whose squared norm contributes to the physics loss.
        """
        raise NotImplementedError


class Poisson1D(PDEOperator):
    """Poisson equation in 1D with known forcing term f(x).

    The PDE is u''(x) = f(x). Here, we define the canonical test with
    analytical solution u(x) = sin(pi x), hence f(x) = -pi^2 sin(pi x).

    Args:
        forcing_fn: Callable producing f(x). If None, uses -pi^2 * sin(pi x).
    """

    def __init__(self, forcing_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        self.forcing_fn = forcing_fn or (lambda x: - (math.pi ** 2) * torch.sin(math.pi * x))

    def residual(self, x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Compute the Poisson residual r(x) = u''(x) - f(x).

        Args:
            x: Collocation points (N, 1) with `requires_grad=True`.
            model: Neural network approximating u(x).
        Returns:
            Residual tensor r(x) with shape (N, 1).
        """
        u = model(x)
        u_xx = grad_n(u, x, n=2)
        f = self.forcing_fn(x)
        return u_xx - f
    
# %% test

if __name__ == '__main__':
    test = True



