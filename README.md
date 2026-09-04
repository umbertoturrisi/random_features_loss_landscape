# Hessian of Loss Landscape and Optimal Learning Strategies in Random Feature Models

Numerical simulations and finite-size scaling analysis of non-convex loss landscapes in overparameterized Teacher-Student architectures, probing Baik-Ben Arous-Péché (BBP) transitions and glassy threshold states.

Based on my Master's Thesis in Physics at Sapienza University of Rome (A.Y. 2025–2026), supervised by Prof. Chiara Cammarota.

---

## Overview

Minimizing high-dimensional non-convex loss functions is a central challenge in machine learning. Standard gradient-based algorithms often become permanently trapped in suboptimal, marginally stable saddles and local minima ("threshold states") characteristic of glassy landscapes.

This repository investigates the **Phase Retrieval** problem under an overparameterized **Random Features** framework:
- Original signal dimension: $D$
- Feature expansion dimension: $N$
- Overparameterization ratio: $\alpha_D = D / N$
- Signal-to-noise ratio: $\alpha = P / N$

Using exact field-theoretic replica calculations (1RSB ansatz) and Random Matrix Theory as theoretical baselines, this codebase provides high-performance JAX simulations to study:
1. **BBP Phase Transitions at Initialization ($\alpha_{\text{BBP}}^{\text{init}}$)**: Spectral detachment of an informative outlier eigenvector aligned with the ground-truth teacher signal.
2. **Glassy Threshold States Instabilities ($\alpha_{\text{BBP}}^{\text{TS}}$)**: Dynamical destabilization of high-energy trapping states along the gradient descent trajectory.
3. **Finite-Size Smearing**: How finite-size scaling ($N < \infty$) blurs strict thermodynamic boundaries, allowing early escape and signal retrieval in theoretically forbidden regimes.
4. **Geometric Constraints**: Comparison between standard spherical normalization and metric-invariant elliptic normalization.

---

## Key Results

<p align="center">
  <img src="figures/Phasediagram.png" width="85%" alt="Empirical vs Theoretical Phase Diagram" />
  <br />
  <em>Figure: Comparison between infinite-size theoretical BBP boundaries (initialization in blue, threshold states in red) and empirical finite-size gradient descent boundaries (green markers), showing significant boundary smearing.</em>
</p>

- **Boundary Blurring**: Finite-size networks ($N \in [100, 5000]$) consistently acquire structural correlation with the signal prior to thermodynamic thresholds.
- **Curvature at Threshold States**: Simulations in a strictly zero-signal setting isolate the local geometry of glassy saddles, verifying that the empirical Hessian spectrum quantitatively matches the continuous bulk predicted by 1RSB replica calculations.

<p align="center">
  <img src="figures/Spectrum.png" width="60%" alt="Hessian Spectrum at Threshold States" />
  <br />
  <em>Figure: Empirical Hessian spectral density at threshold convergence vs. 1RSB theoretical prediction.</em>
</p>

---

## Model Architecture & Loss Formulation

Given sensing vectors $x_\mu \in \mathbb{R}^D$ and a fixed random projection matrix $F \in \mathbb{R}^{D \times N}$, the student predicts:
```math
\hat{y}_{\mu} = \sigma(x_{\mu} F) W
```
where $\sigma(z) = \tanh(z) / c$ is a normalized non-linear activation.

The student minimizes the regularized non-convex loss:
```math
\mathcal{L}(W) = \frac{1}{P} \sum_{\mu=1}^P \frac{(\hat{y}_\mu^2 - y_\mu^2)^2}{a + y_\mu^2}
```

## Setup & Usage

### Prerequisites
- Python $\ge$ 3.9
- CUDA-enabled GPU recommended (handled natively via JAX)

```bash
git clone [https://github.com/umbertoturrisi/random-features-loss-landscape.git](https://github.com/umbertoturrisi/random-features-loss-landscape.git)
cd random-features-loss-landscape
pip install -r requirements.txt
```

### Running Simulations

Run the simulation from the command line:

```bash
python random_features_gd.py --N 1000 --epochs 100000 --eta 0.2 --a 0.01 --norm sphere --sig yes
