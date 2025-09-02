# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:58:09 2025

@author: Altinses
"""

# %% imports

import math
import torch

from typing import Tuple

from config.configuration import Config

from torch.utils.data import DataLoader

from data.collocation_dataset import CollocationDataset
from data.boundary_dataset import BoundaryDataset
from data.supervised_dataset import SupervisedDataset

# %% build data

def u_true(x: torch.Tensor) -> torch.Tensor:
    """Analytical solution u(x) = sin(pi x)."""
    return torch.sin(math.pi * x)

def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for collocation, boundary, and supervised datasets.

    Args:
        cfg: Configuration.
    Returns:
        Tuple of (collocation_loader, boundary_loader, supervised_loader).
    """
    # Collocation points in the open interval (0, 1)
    colloc_ds = CollocationDataset(cfg.n_collocation, cfg.domain, seed=cfg.seed)
    colloc_loader = DataLoader(colloc_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # Dirichlet boundary conditions u(0) = 0, u(1) = 0
    bc_ds = BoundaryDataset({cfg.domain[0]: 0.0, cfg.domain[1]: 0.0})
    bc_loader = DataLoader(bc_ds, batch_size=bc_ds.__len__(), shuffle=False)

    # Optional noisy supervised anchors of the true solution
    sup_ds = SupervisedDataset(
        n_points=cfg.n_supervised,
        fn=u_true,
        noise_std=cfg.noise_std,
        domain=cfg.domain,
        seed=cfg.seed + 1,
    )
    sup_loader = DataLoader(sup_ds, batch_size=min(64, cfg.n_supervised), shuffle=True)

    return colloc_loader, bc_loader, sup_loader


# %% test

if __name__ == '__main__':
    test = True