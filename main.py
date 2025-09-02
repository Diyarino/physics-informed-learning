"""modular_pinn.py

A professional, modular PyTorch template for Physics-Informed Neural Networks (PINNs)
with a simple synthetic dataset. The example solves a 1D Poisson problem on (0, 1):

    u''(x) = -pi^2 * sin(pi * x),    x in (0, 1)
    u(0) = 0,  u(1) = 0

whose analytical solution is:

    u(x) = sin(pi * x)

The code is organized to make it easy to adapt to other PDEs, domains, and
architectures. All components are written with clear interfaces and docstrings.

Author: Diyar Altinses, M.Sc.
License: MIT
"""

# %% imports

from __future__ import annotations
from dataclasses import asdict

import torch
import utils
import data
import model
import config

# %% main

def main() -> None:
    """Train a PINN on the 1D Poisson example and print final losses.

    This function can be adapted for other PDEs by swapping `Poisson1D` with a
    different `PDEOperator` and changing datasets and boundary conditions.
    """
    cfg = config.Config()
    print("Config:", asdict(cfg))
    utils.set_seed(cfg.seed)

    pinn = model.build_pinn(cfg)

    colloc_loader, bc_loader, sup_loader = data.build_dataloaders(cfg)

    optimizer = torch.optim.AdamW(pinn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma)

    trainer = model.Trainer(
        model=pinn,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip=cfg.grad_clip,
    )
    trainer.fit(
        collocation_loader=colloc_loader,
        boundary_loader=bc_loader,
        supervised_loader=sup_loader,
        epochs=cfg.epochs,
        log_every=5,
    )

    # Evaluate on a dense grid for quick sanity check
    with torch.no_grad():
        device = trainer.device
        x_eval = torch.linspace(cfg.domain[0], cfg.domain[1], steps=200).view(-1, 1).to(device)
        pred = pinn.model(x_eval)
        truth = data.u_true(x_eval)
        l2 = torch.sqrt(torch.mean((pred - truth) ** 2)).item()
        print(f"Final L2 error vs analytical solution: {l2:.4e}")

# %% test

if __name__ == "__main__":
    main()
