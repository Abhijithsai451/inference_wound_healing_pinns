# Python Implementation Plan: PINNs for Wound Healing (TRAM)

This document outlines the 30-step roadmap to reimplement the `wound_healing_tram` project using **Physics-Informed Neural Networks (PINNs)** instead of SINDy/FEniCS.

## Phase 1: Deep Learning Setup & Data Ingestion

1.  **Framework Initialization**
    *   Set up PyTorch environment.
    *   Optional: Evaluate libraries like DeepXDE if rapid prototyping is desired, otherwise stick to pure PyTorch for flexibility.

2.  **Tensor Data Loader**
    *   Refactor data handling to convert DataFrames into PyTorch Tensors `(x_tensor, t_tensor, C_tensor)`.
    *   Support batching if datasets are massive (though PINNs often use full-batch for small PDEs).

3.  **Collocation Point Sampler**
    *   Implement a sampler to generate random $(x, t)$ points inside the domain boundaries.
    *   These points are where we enforce the Physics Loss ($L_{PHY}$), even where no data exists.

4.  **Data Normalization (MinMax)**
    *   Implement scaling to map inputs $(x, t)$ and output $C$ to $[0, 1]$ or $[-1, 1]$.
    *   Crucial for Neural Network convergence.

5.  **Synthetic Noise Injector**
    *   Add a utility to purposefully degrade training data.
    *   Objective: Prove PINNs' superiority in handling noise compared to differentiation-based methods (SINDy).

## Phase 2: Neural Network Architecture ($U_{\theta}$)

6.  **MLP Backbone**
    *   Construct the core approximator network (e.g., `feature_dim -> [64, 64, 64, 64] -> 1`).
    *   Use `tanh` or `swish` activations (smooth derivatives are required for calculating gradients).

7.  **Feature Embedding / Input Mapping**
    *   Implement Fourier Feature Embedding if the density has sharp gradients (wound edges).

8.  **Weight Initialization**
    *   Use Xavier (Glorot) initialization.

9.  **Inverse Parameter Layer**
    *   Create a custom `nn.Parameter` class to hold the physics coefficients ($\lambda_{diff}, \lambda_{growth}$).
    *   These are the "unknowns" we solve for.

## Phase 3: Physics-Informed Loss Function ($L_{PHY}$)

10. **Automatic Differentiation Wrapper**
    *   Write a helper `get_gradients(u, x)` utilizing `torch.autograd.grad`.
    *   Must compute higher-order derivatives: $u_t, u_x, u_{xx}$.

11. **PDE Residual Definition**
    *   Define the physics mismatch: $f = u_t - (D \cdot u_{xx} + \rho \cdot u(1-u))$.
    *   This term drives the "Discovery".

12. **Parameter Constraints**
    *   Enforce physical positivity ($D > 0, \rho > 0$).
    *   Implementation: Parametrise as $D = \exp(\hat{D})$ or `softplus(D)`.

13. **Data Loss ($L_{Data}$)**
    *   Standard MSE loss between Network Prediction and Measured Experimental Data.

14. **Boundary Condition Loss ($L_{BC}$)**
    *   Enforce Neumann conditions ($\partial C/\partial x = 0$) at domain boundaries $x=0, x=L$.

15. **Total Loss Aggregation**
    *   $L_{Total} = w_{Data}L_{Data} + w_{PHY}L_{PHY} + w_{BC}L_{BC}$.

## Phase 4: Training & Optimization Logic

16. **Optimizer Strategy (Adam)**
    *   Setup Adam optimizer for the first phase of training (robust global search).

17. **Refinement Strategy (L-BFGS)**
    *   Implement L-BFGS (Quasi-Newton) closure for the second phase.
    *   Critical for finding high-precision parameter values in inverse problems.

18. **Learning Rate Scheduler**
    *   Implement `ReduceLROnPlateau` or Cosine Decay.

19. **Dynamic Loss Balancing**
    *   Implement algorithms (e.g., GradNorm or simple annealing) to adjust weights $w_{Data}$ vs $w_{PHY}$ during training so physics doesn't overpower data (or vice versa).

20. **Training Loop**
    *   The main iteration loop updating both Weights ($\theta$) and Physics params ($\lambda$).

21. **Checkpointing System**
    *   Save model state and current parameter estimates every $N$ epochs.

## Phase 5: Evaluation & Validation

22. **Parameter Convergence Monitor**
    *   Plot the trajectory of discovered parameters ($D, \rho$) over epochs.
    *   Verify they settle to a constant value.

23. **Solution Reconstruction**
    *   Evaluate the trained PINN on a high-resolution grid.
    *   Visualize the "Smoothed / Denoised" Physics Solution.

24. **Residual Field Analysis**
    *   Plot $|f(x,t)|$ across the domain.
    *   High residuals indicate regions where the proposed PDE model fails to capture reality.

25. **Extrapolation / Generalization Test**
    *   Train on time $0 \to T/2$, test prediction accuracy on $T/2 \to T$.

26. **Noise Robustness Benchmark**
    *   Run training sequence on datasets with 1%, 5%, 10% noise and record parameter error.

## Phase 6: Advanced TRAM-Specific Features

27. **Multi-Fidelity Loss**
    *   (Optional) If mixing high-quality and low-quality datasets, assume different noise variances ($\sigma^2$) in the loss.

28. **Sparse Equation Discovery (SINDy-PINN)**
    *   Replace the fixed PDE residual with a library of terms: $f = u_t - \sum \lambda_i \Theta_i$.
    *   Add $L_1$ regularization to $\lambda$ vector to drive trivial terms to zero.

29. **Bayesian PINN (Uncertainty)**
    *   Implement Dropout-based or HMC-based uncertainty estimation for the discovered parameters.
    *   Matches the "Sensitivity Analysis" goal of the original code.

30. **Final Comparative Report**
    *   Generate a document comparing PINN results vs the original FEniCS/SINDy results.
