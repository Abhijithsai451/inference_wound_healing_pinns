import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from logging_utils import setup_logger

logger = setup_logger()


def pivot_data_for_simulation(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pivots the long-form data (Space-Time points) into a (Space x Time) matrix
    required for the forward solver and returns the necessary arrays.
    """
    # Assuming analysis focuses on a single experiment group for validation (e.g., the first one found)
    first_group = df['source_group'].unique()[0]
    df_group = df[df['source_group'] == first_group]

    C_target_df = df_group.pivot(
        index='x_coordinate',
        columns='t_coordinate',
        values=value_col
    ).sort_index(axis=0).sort_index(axis=1)

    x_grid = C_target_df.index.values
    t_eval = C_target_df.columns.values
    C_target_matrix = C_target_df.values  # Shape: (N_x, N_t)

    return x_grid, t_eval, C_target_matrix

class ForwardSolver:
    """
    Implements the 1D PDE Solver by discretizing the SINDy equation into a system of ODEs
    and using scipy.integrate.solve_ivp to solve the system of ODEs.
    """
    def __init__(self,x_grid:np.ndarray, coeffs:np.ndarray, term_names: list, L_domain:float = 1.0):
        """
        Initializes the ForwardSolver based on the discovered SINDy model.
        """
        self.x_grid = x_grid
        self.dx = x_grid[1] -x_grid[0] if len(x_grid) > 1 else L_domain
        self.coeffs = coeffs
        self.term_names = term_names
        self.N_x = len(x_grid)

    def _calculate_derivatives(self, C : np.ndarray)-> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the first and second spatial derivatives of the cell density C using np.gradient.
        However, it assumes there is zero flux (Neumann) boundary Conditions satisfied sufficiently approximates the
        2nd order finite difference derivatives used by np.gradient.
        """
        # Calculate dC/dx (grad_C)
        grad_C = np.gradient(C, self.dx, edge_order=2)

        # Calculate d^2C/dx^2 (laplacian_C)
        laplacian_C = np.gradient(grad_C, self.dx, edge_order=2)

        return grad_C, laplacian_C

    def _rhs_function(self, t: float, C: np.ndarray) -> np.ndarray:
        """
        The right-hand side (RHS) function for solve_ivp.
        Calculates dC/dt = F(C, t) based on the SINDy equation.

        Args:
            t (float): Current time (used by the ODE solver, but the PDE is autonomous).
            C (np.ndarray): Density vector at all spatial points at time t.

        Returns:
            np.ndarray: The temporal derivative vector (dC/dt).
        """
        # Ensure density is physically non-negative
        C = np.maximum(C, 0.0)

        # Calculate spatial derivatives on the fly
        grad_C, laplacian_C = self._calculate_derivatives(C)

        dC_dt = np.zeros_like(C)
        K = 1.0  # Assuming normalized carrying capacity

        # Reconstruct the RHS by summing the weighted candidate terms
        for i, term_name in enumerate(self.term_names):
            xi = self.coeffs[i]

            if np.abs(xi) < 1e-10:  # Skip terms that are pruned to zero
                continue

            term_value = np.zeros_like(C)

            # Map the term name to its calculated value
            if term_name in ['density_gaussian', 'C']:
                term_value = C
            elif term_name == 'grad_C':
                term_value = grad_C
            elif term_name == 'laplacian_C':
                term_value = laplacian_C
            elif term_name == 'C_logistic':
                term_value = C * (1.0 - C / K)
            elif term_name == 'C_pow2':
                term_value = C ** 2
            elif term_name == 'C_pow3':
                term_value = C ** 3
            elif term_name == 'C_gradC':
                term_value = C * grad_C
            elif term_name == 'C_laplacian_C':
                term_value = C * laplacian_C

            dC_dt += xi * term_value

        return dC_dt

    def solve(self, C0: np.ndarray, t_span: tuple, t_eval: np.ndarray):
        """
        Solves the system of ODEs using scipy.integrate.solve_ivp.

        Args:
            C0 (np.ndarray): Initial condition (density profile at t=t_span[0]).
            t_span (tuple): Time range for integration (t_start, t_end).
            t_eval (np.ndarray): Specific time points where the solution is stored.

        Returns:
            scipy.integrate.OdeResult: The result object containing the simulation.
        """
        logger.info(f"Starting forward simulation using solve_ivp (BDF method) for time range {t_span}.")

        result = solve_ivp(
            self._rhs_function,
            t_span,
            C0,
            t_eval=t_eval,
            method='BDF',  # Implicit solver, generally better for stiff PDEs
        )

        if not result.success:
            logger.error(f"Solver failed: {result.message}")

        logger.info(f"Simulation finished. Output shape: {result.y.shape} (Space x Time)")
        return result


class ParameterRefiner:
    """
    Uses scipy.optimize.minimize to fine-tune the SINDy coefficients
    by minimizing the simulation error against target data.
    """

    def __init__(self, x_grid: np.ndarray, all_term_names: list, C_target_all: np.ndarray, t_eval: np.ndarray):
        """
        Args:
            x_grid (np.ndarray): The spatial grid.
            all_term_names (list): All candidate term names (from Theta columns).
            C_target_all (np.ndarray): The target density data (rows=space, cols=time).
            t_eval (np.ndarray): The time points matching the target data columns.
        """
        self.x_grid = x_grid
        self.all_term_names = all_term_names
        self.C_target_all = C_target_all
        self.t_eval = t_eval
        self.t_span = (t_eval[0], t_eval[-1])
        self.C0 = C_target_all[:, 0]  # Initial condition is the first snapshot

    def _cost_function(self, active_coeffs_guess: np.ndarray, initial_coeffs: np.ndarray,
                       active_indices: np.ndarray) -> float:
        """
        The objective function to minimize. Calculates the RMSE between simulation and target data.

        Args:
            active_coeffs_guess (np.ndarray): 1D array of coefficients for *active* terms.
            initial_coeffs (np.ndarray): The full sparse coefficient vector (Xi) from STRidge.
            active_indices (np.ndarray): Indices of the active terms in the full vector.

        Returns:
            float: The Root Mean Square Error (RMSE).
        """
        # 1. Reconstruct the full coefficient vector (Xi)
        current_coeffs = initial_coeffs.copy()
        current_coeffs[active_indices] = active_coeffs_guess.reshape(-1, 1)

        # 2. Instantiate the solver with the new coefficients
        solver = ForwardSolver(
            x_grid=self.x_grid,
            coeffs=current_coeffs,
            term_names=self.all_term_names
        )

        # 3. Run the simulation
        # Only simulate if the initial condition is physically sound
        if np.any(self.C0 < 0):
            logger.warning("Initial condition contains negative values. Aborting cost function.")
            return np.finfo(float).max

        sim_result = solver.solve(self.C0, self.t_span, self.t_eval)

        if not sim_result.success:
            # If the ODE solver fails, return a very high cost
            logger.debug(f"Solver failed in cost function: {sim_result.message}")
            return np.finfo(float).max

        # 4. Calculate RMSE (Root Mean Square Error)
        C_sim = sim_result.y  # C_sim is (Space x Time)

        # Calculate squared error
        squared_error = (self.C_target_all - C_sim) ** 2

        # Calculate RMSE across all space-time points
        rmse = np.sqrt(np.mean(squared_error))

        logger.debug(f"Optimization attempt: RMSE = {rmse:.6f}")
        return rmse

    def refine_coefficients(self, initial_Xi: np.ndarray, method: str = 'L-BFGS-B', **kwargs) -> tuple[
        np.ndarray, float]:
        """
        Performs the optimization of the active SINDy coefficients.

        Args:
            initial_Xi (np.ndarray): The full sparse coefficient vector (Xi) from STRidge.
            method (str): The optimization method to use (e.g., 'L-BFGS-B', 'Nelder-Mead').
            **kwargs: Additional arguments passed to scipy.optimize.minimize.

        Returns:
            tuple[np.ndarray, float]: (Optimized full coefficient vector, Final RMSE).
        """
        logger.info(f"Starting parameter refinement using {method}...")

        # 1. Identify active terms (non-zero coefficients)
        active_indices = np.where(np.abs(initial_Xi.flatten()) > 1e-10)[0]
        initial_active_coeffs = initial_Xi[active_indices].flatten()

        if len(initial_active_coeffs) == 0:
            logger.warning("No active terms found for refinement. Optimization skipped.")
            return initial_Xi, np.nan

        # 2. Prepare fixed arguments for the cost function
        fixed_args = (initial_Xi, active_indices)

        # 3. Run the minimization
        # Use initial active coefficients as the starting guess (x0)
        result = minimize(
            self._cost_function,
            x0=initial_active_coeffs,
            args=fixed_args,
            method=method,
            **kwargs
        )

        # 4. Extract and return the optimized coefficients
        optimized_active_coeffs = result.x
        final_rmse = result.fun

        optimized_Xi = initial_Xi.copy()
        optimized_Xi[active_indices] = optimized_active_coeffs.reshape(-1, 1)

        logger.info(f"Optimization finished. Success: {result.success}. Final RMSE: {final_rmse:.6f}")
        return optimized_Xi, final_rmse