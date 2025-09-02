# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:21:53 2025

@author: Altinses
"""

# %% imports

import torch
import utils

from typing import Optional, Tuple

# %% dataset

class CollocationDataset(torch.utils.data.Dataset):
    """Dataset of interior collocation points for enforcing the PDE residual.

    Args:
        n_points: Number of interior points to sample.
        domain: Tuple (x_min, x_max) specifying the 1D domain.
        seed: Optional seed for reproducible sampling.
    """

    def __init__(self, n_points: int, domain: Tuple[float, float] = (0.0, 1.0), seed: Optional[int] = 42) -> None:
        super().__init__()
        if seed is not None:
            utils.set_seed(seed)
        x = torch.rand(n_points, 1) * (domain[1] - domain[0]) + domain[0]
        x.requires_grad_(True)
        self.x = x

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


# %% test

if __name__ == '__main__':
    dataset = CollocationDataset(100)