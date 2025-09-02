# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:15:41 2025

@author: Altinses
"""

# %% imports

import torch

# %% operators


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute first derivative du/dx via autograd.

    Args:
        outputs: Network outputs u(x) with `requires_grad=True` on inputs.
        inputs: Input coordinates x (N, D).
    Returns:
        Tensor of shape (N, D) representing ∂u/∂x.
    """
    ones = torch.ones_like(outputs)
    (g,) = torch.autograd.grad(outputs, inputs, grad_outputs=ones, create_graph=True)
    return g


def grad_n(outputs: torch.Tensor, inputs: torch.Tensor, n: int) -> torch.Tensor:
    """Compute nth derivative along 1D using autograd.

    Note:
        This helper assumes a 1D input (D=1). For multi-D operators like the
        Laplacian, compose `grad` appropriately.

    Args:
        outputs: Network outputs u(x).
        inputs: Input coordinates x (N, 1) with `requires_grad=True`.
        n: Derivative order (n >= 1).
    Returns:
        Tensor (N, 1) of the nth derivative.
    """
    assert inputs.shape[1] == 1, "grad_n only supports 1D inputs in this template"
    du = outputs
    for _ in range(n):
        du = grad(du, inputs)
    return du



# %% test

if __name__ == '__main__':
    test = True