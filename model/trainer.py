# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:41:02 2025

@author: Altinses
"""

# %% imports

import torch

from model.pinn_wrapper import PINN
from typing import Optional
from utils.device_setting import to_device

# %% traingin


class Trainer:
    """Trainer for PINN models with modular hooks.

    Args:
        model: The `PINN` instance to train.
        optimizer: Torch optimizer.
        scheduler: Optional learning rate scheduler.
        grad_clip: Optional gradient clipping value (L2 norm).
        device: Device to run training on.
    """

    def __init__(
        self,
        model: PINN,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(
        self,
        collocation_loader: torch.utils.data.DataLoader,
        boundary_loader: Optional[torch.utils.data.DataLoader] = None,
        supervised_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 5000,
        log_every: int = 500,
    ) -> None:
        """Train the PINN model.

        Args:
            collocation_loader: DataLoader providing collocation points.
            boundary_loader: Optional DataLoader for boundary points.
            supervised_loader: Optional DataLoader for supervised points.
            epochs: Number of training epochs.
            log_every: Print training progress every `log_every` epochs.
        """
        self.model.train()

        boundary_iter = iter(boundary_loader) if boundary_loader is not None else None
        supervised_iter = iter(supervised_loader) if supervised_loader is not None else None

        for epoch in range(1, epochs + 1):
            for x_col in collocation_loader:
                x_col = to_device(x_col, self.device)

                boundary_batch = None
                if boundary_iter is not None:
                    try:
                        xb, yb = next(boundary_iter)
                    except StopIteration:
                        boundary_iter = iter(boundary_loader)  # type: ignore[arg-type]
                        xb, yb = next(boundary_iter)  # type: ignore[misc]
                    boundary_batch = (to_device(xb, self.device), to_device(yb, self.device))

                supervised_batch = None
                if supervised_iter is not None:
                    try:
                        xs, ys = next(supervised_iter)
                    except StopIteration:
                        supervised_iter = iter(supervised_loader)  # type: ignore[arg-type]
                        xs, ys = next(supervised_iter)  # type: ignore[misc]
                    supervised_batch = (to_device(xs, self.device), to_device(ys, self.device))

                self.optimizer.zero_grad(set_to_none=True)
                losses = self.model.total_loss(x_col, boundary_batch, supervised_batch)
                losses["total"].backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
                lr = self.optimizer.param_groups[0]["lr"]
                printable = {k: float(v.detach().cpu()) for k, v in losses.items()}
                print(f"Epoch {epoch:5d} | lr={lr:.2e} | " + " | ".join([f"{k}={v:.3e}" for k, v in printable.items()]))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the trained model on coordinates `x`.

        Args:
            x: Input coordinates (N, D).
        Returns:
            Predicted solution u(x) with shape (N, 1).
        """
        self.model.eval()
        x = to_device(x, self.device)
        return self.model.model(x)  # type: ignore[attr-defined]

# %% test

if __name__ == '__main__':
    test = True



