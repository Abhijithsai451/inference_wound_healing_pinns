from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from pinn_model import DEVICE, WoundHealingPINN
from wound_healing_tram.pinn_loss_functions import compute_physical_residual

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
    ax1.plot(df_combined['step'], df_combined['L_data'], label='Data Loss ', linestyle='--', color='blue')
    ax1.plot(df_combined['step'], df_combined['L_phy'], label='Physics Loss', linestyle='--', color='red')
    ax1.plot(df_combined['step'], df_combined['L_bc'], label='BC Loss', linestyle='--', color='green')

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

    ax2.plot(df_combined['step'], df_combined['D'], label='Diffusivity', color='darkorange', linewidth=2)
    ax2.plot(df_combined['step'], df_combined['rho'], label='Proliferation', color='purple', linewidth=2)

    ax2.axvline(x=start_step, color='gray', linestyle=':', label='Adam $\to$ L-BFGS Transition')

    ax2.set_xlabel('Training Step (Epoch/Iteration)')
    ax2.set_ylabel('Discovered Parameter Value')
    ax2.set_title('Discovered Physics Parameters Trajectory')
    ax2.legend(loc='upper right')
    ax2.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(Path(PLOT_SAVE_PATH) / f'{output_filename}_params.png')
    plt.close(fig)

    print(f"\n Training convergence plots saved to: {Path(PLOT_SAVE_PATH).resolve()}")


def generate_prediction_grid(model: WoundHealingPINN, scaler: MinMaxScaler,
                             resolution: int = 50) -> pd.DataFrame:
    """
    Generates a high-resolution spatio-temporal grid, runs the trained PINN model on it, and denormalizes the results.
    """
    # 1. Define High-Resolution Normalized Grid [0, 1]
    # Create 1D arrays for each dimension
    x_normalized = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    y_normalized = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    t_normalized = np.linspace(0.0, 1.0, resolution, dtype=np.float32)

    # Create a 3D meshgrid
    X_mesh, Y_mesh, T_mesh = np.meshgrid(x_normalized, y_normalized, t_normalized, indexing='ij')

    # Flatten and combine into a tensor: (N_total, 3) where N_total = resolution^3
    X_input_np = np.stack([X_mesh.flatten(), Y_mesh.flatten(), T_mesh.flatten()], axis=1)

    # Convert to tensor and move to device
    X_input_tensor = torch.tensor(X_input_np, dtype=torch.float32).to(DEVICE)

    # 2. Evaluate Trained PINN
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        C_hat_normalized_tensor = model(X_input_tensor).cpu().numpy()
    model.train()  # Set model back to training mode

    # 3. Denormalize the Inputs (x, y, t) and Output (C_hat)

    # The scaler was fitted on (x, y, t, C) data. We mimic this structure for denormalization.
    X_dummy_output = np.concatenate([X_input_np, C_hat_normalized_tensor], axis=1)

    # Inverse transform everything at once to get physical scales
    X_denormalized = scaler.inverse_transform(X_dummy_output)

    # 4. Create Output DataFrame
    df_pred = pd.DataFrame({
        'x': X_denormalized[:, 0],
        'y': X_denormalized[:, 1],
        't': X_denormalized[:, 2],
        'C_pred': X_denormalized[:, 3]
    })

    print(f"Solution reconstructed on a {resolution}x{resolution}x{resolution} grid ({len(df_pred)} points).")
    return df_pred


def analyze_residual_field(model: WoundHealingPINN, scaler: MinMaxScaler, df_pred: pd.DataFrame,
                           resolution: int = 50) -> pd.DataFrame:
    """
    Calculates the absolute PDE residual |f(x,t)| on the reconstructed prediction grid.
    """
    print("\nCalculating PDE Residual Field...")

    # 1. Prepare normalized input tensor from the prediction grid coordinates
    X_input_np = df_pred[['x', 'y', 't']].values

    X_normalized_coords = scaler.transform(np.concatenate([X_input_np, np.zeros((len(X_input_np), 1))], axis=1))[:, :3]

    X_input_tensor = torch.tensor(X_normalized_coords, dtype=torch.float32).to(DEVICE)

    dX_scale = scaler.data_max_[0] - scaler.data_min_[0]
    dY_scale = scaler.data_max_[1] - scaler.data_min_[1]
    dT_scale = scaler.data_max_[2] - scaler.data_min_[2]
    dC_scale = scaler.data_max_[3] - scaler.data_min_[3]

    # 2. Compute the physical residual (f_phys)
    model.eval()
    with torch.enable_grad():  # Autograd requires gradient tracking, even in eval mode
        C_t_phys, C_hat, C_xx_phys, C_yy_phys = compute_physical_residual(
            model,
            X_input_tensor,
            dC_scale, dX_scale, dY_scale, dT_scale
        )
        # 2. Calculate the specific residual based on the current mode
        D, rho = model.pde_params
        f_phys = (C_t_phys -
                  D * (C_xx_phys + C_yy_phys) -
                  rho * C_hat.squeeze() * (1 - C_hat.squeeze()))

        # 3. Convert to absolute numpy array for the dataframe
        residual_values = torch.abs(f_phys).detach().cpu().numpy()
    model.train()
    f_phys = f_phys.detach().cpu().numpy()
    # 3. Store the absolute residual
    df_pred['Residual'] = np.abs(f_phys)

    print(f"Residual field calculated. Max Residual: {df_pred['Residual'].max():.4e}")
    return df_pred


def plot_spatial_residuals(df_res: pd.DataFrame, output_dir: str = "plots/residuals"):
    """
    Creates heatmaps of the absolute PDE residual |f(x,t)| at various time steps.
    High residual areas indicate where the model is not capturing the physics accurately.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Select a few representative time points (e.g., Start, Middle, End)
    unique_times = np.sort(df_res['t'].unique())
    indices = np.linspace(0, len(unique_times) - 1, 4, dtype=int)
    selected_times = unique_times[indices]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, t_val in enumerate(selected_times):
        # Filter data for the specific time slice
        subset = df_res[df_res['t'] == t_val]

        # Pivot for heatmap plotting
        pivot_table = subset.pivot(index='y', columns='x', values='Residual')

        sns.heatmap(pivot_table, ax=axes[i], cmap='viridis', cbar_kws={'label': '|f(x,y,t)|'})
        axes[i].set_title(f"Residual at t = {t_val:.1f}h")
        axes[i].invert_yaxis()  # Match spatial orientation

    plt.tight_layout()
    save_path = Path(output_dir) / "spatial_residual_evolution.png"
    plt.savefig(save_path)
    print(f"Spatial residual heatmaps saved to: {save_path}")
    plt.show()