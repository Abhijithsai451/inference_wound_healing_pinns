import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from logging_utils import setup_logger

logger = setup_logger()
RESULTS_DIR = "wound_healing_jin/results"
def calculate_rmse(C_sim: np.ndarray, C_target: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Error (RMSE) between simulated and target data.

    Args:
        C_sim (np.ndarray): Simulated density data (Space x Time).
        C_target (np.ndarray): Target/Experimental density data (Space x Time).

    Returns:
        float: The RMSE value.
    """
    if C_sim.shape != C_target.shape:
        logger.error(f"Shape mismatch: Simulation {C_sim.shape} vs Target {C_target.shape}")
        return np.nan

    squared_error = (C_target - C_sim)**2
    rmse = np.sqrt(np.mean(squared_error))
    logger.info(f"Calculated RMSE: {rmse:.6f}")
    return rmse

def plot_density_evolution(
    x_grid: np.ndarray,
    t_eval: np.ndarray,
    C_sim: np.ndarray,
    C_target: np.ndarray,
    rmse: float,
    equation_str: str,
    output_filename: str = 'wound_healing_jin/validation_plot.png'
):
    """
    Generates a plot comparing simulated density profiles against experimental data
    at a few select time points.

    Args:
        x_grid (np.ndarray): Spatial grid points.
        t_eval (np.ndarray): Time points where data is evaluated/simulated.
        C_sim (np.ndarray): Simulated density data (Space x Time).
        C_target (np.ndarray): Target/Experimental density data (Space x Time).
        rmse (float): The calculated RMSE for the plot title.
        equation_str (str): The discovered PDE for the plot title.
        output_filename (str): Path and name for saving the plot.
    """
    logger.info(f"Generating validation plot: {output_filename}")

    # 1. Select key time indices to plot (e.g., beginning, middle, end)
    # Plot 4 snapshots: 0%, 33%, 66%, 100% of the total time points
    time_indices = np.linspace(0, len(t_eval) - 1, 4, dtype=int)

    # 2. Setup the plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    # Clean up equation string for LaTeX/title use
    title_eq = equation_str.replace('\\frac{\\partial C}{\\partial t} \\approx ', '')
    title_eq = title_eq.replace('\\nabla C', 'C_x').replace('\\nabla^2 C', 'C_{xx}')

    fig.suptitle(f"SINDy Model Validation (RMSE: {rmse:.4e})\n$dC/dt \\approx {title_eq}$", fontsize=12)

    # 3. Plot each snapshot
    for i, t_idx in enumerate(time_indices):
        ax = axes[i]
        t = t_eval[t_idx]

        # Experimental Data (Markers)
        ax.plot(x_grid, C_target[:, t_idx], 'o', color='gray', alpha=0.5, label='Experimental Data')

        # Simulation Results (Line)
        ax.plot(x_grid, C_sim[:, t_idx], '-', color='crimson', linewidth=2, label='SINDy Simulation')

        # Formatting
        ax.set_title(f'Time $t={t:.2f}$', fontsize=10)
        ax.set_xlabel('Spatial Coordinate $x$', fontsize=10)

        # Set shared Y-label only on the first axis
        if i == 0:
            ax.set_ylabel('Density $C(x, t)$', fontsize=10)
            ax.legend(fontsize=8, loc='lower left')

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Save the figure
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    plt.savefig(os.path.join(RESULTS_DIR, output_filename))
    plt.close(fig)
    logger.info(f"Plot saved to results/{output_filename}")