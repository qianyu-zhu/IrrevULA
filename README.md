# Irreversible Perturbations for MCMC Sampling

This repository contains code for studying irreversible perturbations in Markov Chain Monte Carlo (MCMC) sampling methods, specifically focusing on Langevin dynamics with various perturbation strategies.

## Overview

We investigate how adding irreversible (skew-symmetric) perturbations to the diffusion matrix in Langevin dynamics can accelerate convergence and reduce asymptotic variance. The code compares several perturbation strategies:

- **Unperturbed**: Standard reversible Langevin dynamics
- **Irreversible-S/M/L**: Small, medium, and large random perturbations
- **Irreversible-O**: Optimal perturbation (maximizes asymptotic variance reduction)
- **Irreversible-SO**: Spectral-optimal perturbation

## Experiments

### Experiment 1: Ellipse (4D Gaussian)
`exp1_ellipse/` - Sampling from a 4-dimensional Gaussian with diagonal covariance. Tests convergence under fixed step budget and fixed trajectory length.

### Experiment 2: Multi-modal Gaussian Mixture
`exp2_multi_gaussian/` - Sampling from a 3-dimensional Gaussian mixture model. Evaluates mixing between modes.

### Experiment 3: Bayesian Logistic Regression
`exp3_manifold_blr/` - Bayesian inference on logistic regression using Riemannian manifold Langevin dynamics with UCI benchmark datasets.

### Experiment 4: Independent Component Analysis
`exp4_ICA/` - Posterior sampling for ICA model parameters.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- tqdm
- joblib

Install dependencies:
```bash
pip install numpy scipy matplotlib tqdm joblib
```

## Project Structure

```
.
├── exp1_ellipse/
│   ├── helper.py                      # Core functions for perturbation construction
│   ├── simulate_fixed_steps.py        # Fixed step budget experiments
│   ├── simulate_fixed_time.py         # Fixed trajectory length experiments
│   ├── simulate_adaptive_*.py         # Adaptive methods
│   └── *.ipynb                        # Analysis notebooks
├── exp2_multi_gaussian/
│   ├── helper.py
│   ├── simulate_*.py
│   └── *.ipynb
├── exp3_manifold_blr/
│   ├── helper.py
│   ├── benchmarks.mat                 # UCI benchmark datasets
│   ├── simulate_blr_*.py
│   └── fixbudget.ipynb
├── exp4_ICA/
│   ├── helper.py
│   ├── ica_data3.mat                  # ICA experiment data
│   ├── simulate_ica_*.py
│   └── integration.py
└── README.md
```

## Usage

Each experiment folder contains simulation scripts that can be run independently:

```bash
cd exp1_ellipse
python simulate_fixed_steps.py --h 0.01 --K 100000 --M 128
```

Key parameters:
- `--h`: Step size (discretization parameter)
- `--K`: Number of MCMC steps
- `--M`: Number of parallel chains
- `--T`: Total simulation time (for fixed-time experiments)

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]
