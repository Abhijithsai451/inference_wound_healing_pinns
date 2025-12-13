# Python Implementation Plan for Wound Healing (TRAM)

This document outlines the 30-step roadmap to port the `wound_healing_tram` FEniCS workflow into a pure Python scientific stack.

## Phase 1: Advanced Configuration & Data Management

1.  **Global Settings Module**
    *   Create `config_manager.py` that loads and validates settings found in `vsiParams.py` and `Step0_makeVSIParams.py`.
    *   Support "parameter sets" (combinations of smoothing windows, grid sizes) to allow batch processing.

2.  **Experiment Manifest Builder**
    *   Write a script to scan the `data` folder (2600+ files) and build a structured `manifest.csv`.
    *   Extract metadata: `initCells`, `drugConcentration` (if applicable), `replicateID`, and `timestamp`.

3.  **Data Ingestion Class**
    *   Implement a `TramDataLoader` class that uses the manifest to load specific experimental groups.
    *   Ensure robust error handling for missing or corrupted files.

4.  **Advanced Smoothing (Gaussian + Moving Average)**
    *   Implement the exact smoothing logic from `Step1_read_write.py`.
    *   Likely involves a specific sequence of temporal rolling means followed by spatial Gaussian filtering.

5.  **Grid Interpolation Utility**
    *   Create a function to interpolate experimental data (irregular or pixel-based) onto a uniform Finite Difference grid (`dx`) required for simulations.

6.  **Data Validation Suite**
    *   Write unit tests to detect data anomalies: NaNs, infinite values, or non-physical negative densities.

## Phase 2: Enhanced Feature Engineering

7.  **Finite Difference Stencils**
    *   Implement high-order finite difference coefficients (stencil size 5 or 7) for `dC/dx` and `d^2C/dx^2`.
    *   Goal: Minimize numerical error on the spatial grid.

8.  **Physics Kernel Library**
    *   Create a library of functions mimicking `Step2_generate_basis.py`.
    *   Terms: Diffusion ($\nabla^2 C$), Advection ($\nabla C$), Logistic Growth ($C(1-C)$), and Chemotaxis logic.

9.  **Interaction Terms Generator**
    *   Implement logic to analytically generate cross-terms (e.g., $C \cdot \nabla C$, $C^2 \nabla C$) from a list of primitive features.

10. **Design Matrix Assembler ($\Theta$)**
    *   Write a function to construct the $\Theta$ matrix.
    *   Structure: Columns = Physics Terms, Rows = Space-Time Points.

11. **Feature Scaling/Normalization**
    *   Implement a standardizer (e.g., `StandardScaler`) for columns of $\Theta$.
    *   Critical for regression stability when mixing small diffusion scales with large growth scales.

12. **Time Derivative Calculator**
    *   Implement a robust `dC/dt` calculator.
    *   Consider using Spline Interpolation derivatives instead of raw finite differences to handle noise in time series.

## Phase 3: Variable Selection & Model Discovery

13. **Stepwise Regression Engine**
    *   Port logic from `stepwiseRegression.py` to a Python class.
    *   Core mechanism: Backward Elimination (start full, drop terms one by one).

14. **Ridge Regression Subroutine**
    *   Implement `Ridge` solver with Cross-Validation (CV) to auto-select optimal regularization strength ($\alpha$) within the loop.

15. **F-Test Implementation**
    *   Implement the statistical F-test used in the original code.
    *   Logic: Only drop a term if the increase in error is statistically insignificant.

16. **Group Selection Logic**
    *   Port the "Grouping" feature from `Step3_linear_VSI.py`.
    *   Rules: Certain terms (e.g., advective group) must be dropped or kept together to maintain physical consistency.

17. **Model Stability Check**
    *   Add post-discovery validation.
    *   Example: Diffusion coefficient must be positive. If negative, reject/penalize the model.

18. **Equation Stringifier**
    *   Create a formatter to output the discovered SINDy model as a human-readable LaTeX equation.

19. **VSI Reporting**
    *   Generate a "Discovery Report" tracking which terms were dropped at which step and why (F-score vs Threshold).

## Phase 4: Optimization & Refinement

20. **ODE System Generator**
    *   Write a function converting the discovered spatial PDE into a system of ODEs ($dC_i/dt = f(C_i, ...)$).

21. **Forward Solver (IVP)**
    *   Implement time-stepping using `scipy.integrate.solve_ivp`.
    *   Configuration: Must use implicit methods (e.g., `BDF`, `Radau`) for stiff PDE systems.

22. **Loss Function Definition**
    *   Define $J(\theta) = ||C_{sim} - C_{data}||^2 + \lambda ||\theta||^2$.

23. **Parameter Bounds Manager**
    *   Map discovered parameters to physical limits (e.g., $D \in [0, D_{max}]$).
    *   Prevents optimizer from drifting into non-physical regimes.

24. **Adjoint-Like Optimization Loop**
    *   Implement the minimization routine using `scipy.optimize.minimize` (L-BFGS-B) to refine the coefficients found in Phase 3.

## Phase 5: Sensitivity Analysis & Validation

25. **Sensitivity Matrix Calculator**
    *   Port `Step6a_SensitivityData.py`.
    *   Method: Perturb each parameter by $\pm 1\%$ and measure output deviation to quantify parameter importance.

26. **Confidence Interval Estimation**
    *   Calculate uncertainty intervals for parameters based on regression residuals and Hessian approximation.

27. **Forward Prediction Validator**
    *   Implement "Hold-out" testing: Optimize on $t=0..t_{half}$, predict $t_{half}..t_{end}$ to assess generalization.

28. **Residual Analysis**
    *   Visualize $C_{data} - C_{model}$ heatmaps.
    *   Goal: Identify systematic spatial or temporal biases (e.g., errors concentrated at wound edges).

29. **Sensitivity Plotter**
    *   Port `Step6b_PlotSensitivityData.py` to create Tornado plots or Bar charts of parameter sensitivity indices.

30. **Final Report Generator**
    *   Create a clean summary output including:
        *   Data smoothing verification.
        *   The Discovered Equation.
        *   Goodness-of-Fit plots.
        *   Sensitivity Analysis results.
