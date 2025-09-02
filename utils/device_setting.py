# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:07:42 2025

@author: Altinses
"""

# %% imports

import torch

# %% functions

def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to a given device, preserving `requires_grad`.

    Args:
        x: Input tensor.
        device: Target device.
    Returns:
        Tensor on the specified device.
    """
    return x.to(device)


# %% test

if __name__ == '__main__':
    tensor = torch.rand(16,8)
    tensor_device = to_device(tensor)