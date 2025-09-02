# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:04:54 2025

@author: Altinses
"""

# %% imports

import random
import torch

# %% function

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy (if available), and PyTorch.

    Args:
        seed: The seed value to make experiments reproducible.
    """
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    
# %% test

if __name__ == '__main__':
    set_seed()