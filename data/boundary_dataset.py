# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:26:51 2025

@author: Altinses
"""

# %% imports

import torch

from typing import Dict, Tuple

# %% dataset class

class BoundaryDataset(torch.utils.data.Dataset):
    """Dataset for boundary condition points (Dirichlet in this example).

    Args:
        values: A dictionary mapping scalar boundary position -> u(boundary).
               Example: {0.0: 0.0, 1.0: 0.0}
    """

    def __init__(self, values: Dict[float, float]) -> None:
        super().__init__()
        xs = torch.tensor([[k] for k in values.keys()], dtype=torch.float32)
        ys = torch.tensor([[v] for v in values.values()], dtype=torch.float32)
        xs.requires_grad_(True)
        self.x = xs
        self.y = ys

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
    
    
# %% test

if __name__ == '__main__':
    test = True