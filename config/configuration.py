# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:44:27 2025

@author: Altinses
"""

# %%

from dataclasses import dataclass
from typing import Tuple, Optional

# %% config

@dataclass
class Config:
    """Configuration for training and data generation."""

    # Data
    domain: Tuple[float, float] = (0.0, 1.0)
    n_collocation: int = 2048
    n_supervised: int = 32
    noise_std: float = 0.01

    # Model
    in_dim: int = 1
    out_dim: int = 1
    hidden_layers: int = 5
    hidden_units: int = 64
    activation: str = "tanh"  # choices: "tanh", "gelu", "relu"
    use_weight_norm: bool = False

    # Loss Weights
    w_physics: float = 1.0
    w_boundary: float = 10.0
    w_supervised: float = 0.5

    # Optimization
    epochs: int = 4000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    scheduler_step: int = 2000
    scheduler_gamma: float = 0.5

    # Reproducibility
    seed: int = 42
    
    
# %% test