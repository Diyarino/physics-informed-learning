# Physics-Informed Neural Networks (PINN) — Modular PyTorch Template

This repository provides a **clean, modular, and extensible PyTorch template** for building and training Physics-Informed Neural Networks (PINNs).  
The code is structured into clear modules for configuration, data handling, models, utilities, and training, making it easy to adapt to **different PDEs, boundary conditions, and problem domains**.

The included example solves a **1D Poisson problem** on the unit interval:

$$
u''(x) = -\pi^2 \sin(\pi x), \quad x \in (0, 1), \quad u(0) = u(1) = 0,
$$

whose analytical solution is:

$$
u(x) = \sin(\pi x).
$$

---

## 📂 Project Structure

.
├── config/
│ ├── init.py
│ └── configuration.py # Central configuration (hyperparameters, model setup)
│
├── data/
│ ├── init.py
│ ├── boundary_dataset.py # Boundary condition dataset
│ ├── collocation_dataset.py # Interior collocation dataset
│ ├── supervised_dataset.py # Optional supervised data (e.g. noisy measurements)
│ └── build_dataloaders.py # Helper to construct DataLoaders
│
├── model/
│ ├── init.py
│ ├── build_pinn.py # Build PINN wrapper from config
│ ├── feedforward.py # Generic feedforward network (MLP)
│ ├── pinn_wrapper.py # Combines network + PDE residual into PINN
│ └── trainer.py # Training loop for PINNs
│
├── utils/
│ ├── init.py
│ ├── device_setting.py # Device management (CPU/GPU)
│ ├── diff_operators.py # Autograd-based differential operators
│ ├── physics_pde_operators.py # PDE residual operators (e.g. Poisson1D)
│ └── seed_setting.py # Reproducibility utilities
│
├── main.py # Entry point: trains and evaluates a PINN
└── README.md



---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/diyarino/physics-informed-learning.git
cd physics-informed-learning
```

### 2. Install dependencies 

```bash
pip install torch numpy
```

(Optional: use a virtual environment or conda environment for isolation.)

### 3. Run the example

```bash
python main.py
```

---

## ⚙️ Configuration

All hyperparameters (training setup, model architecture, loss weights, etc.) are stored in
[`config/configuration.py`](./config/configuration.py).

Example:

```python
# config/configuration.py
in_dim = 1
out_dim = 1
hidden_layers = 5
hidden_units = 64
activation = "tanh"

n_collocation = 2048
n_supervised = 32
epochs = 4000
lr = 1e-3
```

---

## 📊 Extending to Other PDEs

1. **Define a PDE residual operator**
   Add a new class to [`utils/physics_pde_operators.py`](./utils/physics_pde_operators.py)
   by subclassing `PDEOperator` and implementing `residual(x, model)`.

2. **Adjust boundary/supervised datasets**
   Modify or create new datasets in the `data/` folder.

3. **Update config**
   Change `configuration.py` to set training parameters and loss weights.

4. **Run training**
   Launch `main.py` again.

---

## 📈 Example Output

During training, you’ll see loss logs:

```
Epoch   500 | lr=1.00e-03 | physics=2.34e-03 | boundary=1.12e-06 | supervised=8.23e-05 | total=2.42e-03
...
Final L2 error vs analytical solution: 2.51e-03
```

---

## 🧩 Key Features

* Modular design for **easy adaptation** to new PDEs
* Supports **physics, boundary, and supervised losses**
* Clean separation of **config, data, model, and utils**
* Reproducible experiments with **seed management**
* Extendable training loop with **schedulers and gradient clipping**

---

## 📜 License

This project is licensed under the MIT License.
See [LICENSE](./LICENSE) for details.












