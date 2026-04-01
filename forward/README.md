# Case Folder Structure

Each case folder in this repository follows a common structure. Below is a general guide to the files and subdirectories you will find inside any case.

## How to Run

**Most cases** are executed via Slurm on an HPC cluster:

```bash
cd forward/case_folder_name/
./bash.sh
```

**Demonstration notebook:** The `forward/1d_poisson_b50_dom512/` case includes a `demo_local_1trial.ipynb` Jupyter notebook that walks through the PECANN-CAPU workflow. This is the recommended starting point for new users who want to understand the algorithm before running full-scale cases.

## General Directory Structure

```
case_folder/
├── config.py              # Configuration and hyperparameters
├── main.py                # Main training script
├── run.slurm              # Slurm job submission script
├── bash.sh                # Clean previous outputs and submit Slurm job
├── core/
│   └── trainer.py         # Training loop and ALM framework
├── data/
│   └── collocation*.py    # Collocation point sampling
├── diagnostics/
│   └── diagnostics.py     # Diagnostic utilities
├── models/
│   └── fc_net.py          # Neural network architecture
├── physics/
│   └── residuals.py       # PDE residuals and boundary/initial conditions, etc.
├── utils/
│   ├── io.py              # I/O utilities
│   └── layers.py          # Custom layers (e.g. Fourier features)
├── viz/
│   └── viz.py             # Visualization utilities
├── logs/                  # Training logs and saved model checkpoints
│   ├── *_constr.dat       # Constraint history
│   ├── *_object.dat       # Objective history
│   ├── *_l2.dat           # Relative l2 error history
│   ├── *_linf.dat         # linf error history
│   ├── *_lambda.dat       # Lagrange multiplier history
│   ├── *_mu.dat           # Penalty parameter history
│   └── *.pt               # Saved model checkpoints
└── pic/                   # Output figures
    └── *.png              # Loss curves, solution plots, etc.
```

## Key File Descriptions

**`config.py`** — Defines all problem-specific settings: domain, boundary conditions, network size, optimizer choice, learning rate schedule, number of collocation points, and PECANN-CAPU penalty scaling factors (ηs).

**`core/trainer.py`** — Assembles the model, evaluation function, and PECANN-CAPU trainer, then runs the training loop, calls the primal optimizer and dual update, and logs training metrics. All cases import the core algorithm via:

```python
from pecann_capu import PECANNTrainer, PECANNState
```

**`physics/residuals.py`** — Defines the PDE residuals and boundary/initial condition losses specific to each problem. This is the main file that changes between cases.

**`models/fc_net.py`** — Neural network architecture definition. Most cases use a fully connected network, optionally with Fourier feature layers.

## Demo Notebook

The `forward/1d_poisson_b50_dom512/demo_local_1trial.ipynb` notebook provides an interactive walkthrough of the PECANN-CAPU algorithm applied to the 1D Poisson equation with a multi-scale solution.
