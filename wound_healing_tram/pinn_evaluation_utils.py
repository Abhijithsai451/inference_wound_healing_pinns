from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

PLOT_SAVE_PATH = 'plots'

def plot_training_convergence(adam_history: List[Dict], lbfgs_history: List[Dict], output_filename: str = 'convergence_plots'):
    """
    PLots the Total loss, individual loss components and parameter trajjectory from the combined Adam and L-BFGS optimizers.
    """
    # Convert Adam history to DataFrame, using 'epoch' as the step index
    df_adam = pd.DataFrame(adam_history)
    df_adam['step'] = df_adam['epoch']

    # Convert L-BFGS history to DataFrame, offset the step index
    df_lbfgs = pd.DataFrame(lbfgs_history)
    start_step = df_adam['step'].max() if not df_adam.empty else 0
    # Add a small buffer of 10 steps for visual separation between phases
    df_lbfgs['step'] = df_lbfgs['iter'] + start_step + 10

    # Rename 'iter' column to 'epoch' for consistency in L-BFGS
    df_lbfgs.rename(columns={'iter': 'epoch'}, inplace=True)

    # Combine the two dataframes
    df_combined = pd.concat([df_adam, df_lbfgs], ignore_index=True)

    # --- 2. Create Plots ---

    # Create the directory if it doesn't exist
    Path(PLOT_SAVE_PATH).mkdir(exist_ok=True)

    # --- Plot A: Loss History ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df_combined['step'], df_combined['L_total'], label='Total Loss', color='black', linewidth=2)
    ax1.plot(df_combined['step'], df_combined['L_data'], label='Data Loss ($L_{Data}$)', linestyle='--', color='blue')
    ax1.plot(df_combined['step'], df_combined['L_phy'], label='Physics Loss ($L_{Phy}$)', linestyle='--', color='red')
    ax1.plot(df_combined['step'], df_combined['L_bc'], label='BC Loss ($L_{BC}$)', linestyle='--', color='green')

    # Vertical line to separate training phases
    ax1.axvline(x=start_step, color='gray', linestyle=':', label='Adam $\to$ L-BFGS Transition')

    ax1.set_xlabel('Training Step (Epoch/Iteration)')
    ax1.set_ylabel('Loss Value (Log Scale)')
    ax1.set_title('PINN Training Loss Convergence')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(Path(PLOT_SAVE_PATH) / f'{output_filename}_loss.png')
    plt.close(fig)

    # --- Plot B: Parameter Trajectory ---
    fig, ax2 = plt.subplots(figsize=(12, 6))

    ax2.plot(df_combined['step'], df_combined['D'], label='Diffusivity ($D$)', color='darkorange', linewidth=2)
    ax2.plot(df_combined['step'], df_combined['rho'], label='Proliferation ($\rho$)', color='purple', linewidth=2)

    ax2.axvline(x=start_step, color='gray', linestyle=':', label='Adam $\to$ L-BFGS Transition')

    ax2.set_xlabel('Training Step (Epoch/Iteration)')
    ax2.set_ylabel('Discovered Parameter Value')
    ax2.set_title('Discovered Physics Parameters Trajectory')
    ax2.legend(loc='upper right')
    ax2.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(Path(PLOT_SAVE_PATH) / f'{output_filename}_params.png')
    plt.close(fig)

    print(f"\n✅ Training convergence plots saved to: {Path(PLOT_SAVE_PATH).resolve()}")
