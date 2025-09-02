# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:28:57 2025

@author: Altinses
"""

# %% imports

import torch

from utils.seed_setting import set_seed 
from typing import Callable, Optional, Tuple

# %% dataset class


class SupervisedDataset(torch.utils.data.Dataset):
    """Optional supervised points to anchor the solution (noisy measurements).

    Args:
        n_points: Number of supervised samples.
        fn: Ground-truth function u(x) for generating synthetic labels.
        noise_std: Standard deviation of Gaussian noise added to labels.
        domain: Tuple (x_min, x_max) of the 1D interval.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        n_points: int,
        fn: Callable[[torch.Tensor], torch.Tensor],
        noise_std: float = 0.0,
        domain: Tuple[float, float] = (0.0, 1.0),
        seed: Optional[int] = 123,
    ) -> None:
        super().__init__()
        if seed is not None:
            set_seed(seed)
        x = torch.rand(n_points, 1) * (domain[1] - domain[0]) + domain[0]
        with torch.no_grad():
            y = fn(x)
            if noise_std > 0:
                y = y + noise_std * torch.randn_like(y)
        x.requires_grad_(True)
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

# %% test

if __name__ == '__main__':
    test = True