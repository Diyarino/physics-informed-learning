# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:48:00 2025

@author: Altinses
"""

# %% imports

import math
import torch

from typing import Callable

from utils.physics_pde_operators import Poisson1D
from config.configuration import Config
from model.pinn_wrapper import PINN, LossWeights
from model.feedforward import FeedForwardNN

# %% build model

def f_poisson(x: torch.Tensor) -> torch.Tensor:
    """Forcing term f(x) = -pi^2 * sin(pi x) consistent with u''(x) = f(x)."""
    return - (math.pi ** 2) * torch.sin(math.pi * x)

def get_activation(name: str) -> Callable[[], torch.nn.Module]:
    """Factory for activation functions.

    Args:
        name: Name of the activation ("tanh", "gelu", "relu").
    Returns:
        A callable that instantiates the activation module.
    """
    name = name.lower()
    if name == "tanh":
        return torch.nn.Tanh
    if name == "gelu":
        return torch.nn.GELU
    if name == "relu":
        return torch.nn.ReLU
    raise ValueError(f"Unsupported activation: {name}")

def build_pinn(cfg: Config) -> PINN:
    """Construct a PINN instance from configuration.

    Args:
        cfg: Configuration object.
    Returns:
        Initialized `PINN` ready for training.
    """
    act = get_activation(cfg.activation)
    model = FeedForwardNN(
        in_dim=cfg.in_dim,
        out_dim=cfg.out_dim,
        hidden_layers=cfg.hidden_layers,
        hidden_units=cfg.hidden_units,
        activation=act,
        use_weight_norm=cfg.use_weight_norm,
    )
    pde = Poisson1D(forcing_fn=f_poisson)
    lw = LossWeights(cfg.w_physics, cfg.w_boundary, cfg.w_supervised)
    return PINN(model, pde, lw)