from pathlib import Path
from typing import Dict
import pandas as pd
import torch

from pinn_data_preprocessing import import_data
from logging_utils import setup_logger
from wound_healing_tram.pinn_evaluation_utils import (
    plot_training_convergence,
    generate_prediction_grid,
    analyze_residual_field, plot_spatial_residuals
)
from wound_healing_tram.pinn_loss_functions import PINNLoss
from wound_healing_tram.pinn_model import WoundHealingPINN, USE_FOURIER_FEATURES, get_device
from wound_healing_tram.pinn_trainer import PINNTrainer
from wound_healing_tram.tensor_data_utils import (
    sample_collocation_points,
    noise_injection,
    sample_boundary_points,
    convert_to_tensors
)
from wound_healing_tram.visualize_data import visualize_plots

# Initialization
logger = setup_logger()
DEVICE = get_device()

# %% Hyper Parameters
NUM_COLLOCATION_POINTS = 500
NUM_BOUNDARY_POINTS = 100
NOISE_STD = 0.01
EPOCHS = 30
LBFGS_ITER = 30
LR = 1e-3
GRID_RESOLUTION = 50
NOISE_LEVELS = [0.00, 0.05, 0.10]

# Loss Weights
W_DATA = 100.0
W_PHY = 1.0
W_BC = 10.0

SINDY_MODE = True  # Set to True to discover the equation structure
LAMBDA_SINDY = 1e-4 if SINDY_MODE else 0.0  # L1 Sparsity penalty


def run_pinn_experiment(noise_std: float, total_data: pd.DataFrame) -> Dict:
    """
    Runs the full PINN training and evaluation pipeline for a given noise level.
    """
    logger.info(f"\n[BENCHMARK] Starting Experiment: Noise {noise_std * 100:.1f}%")

    # 1. Prepare Tensors
    X_data, C_data, scaler = convert_to_tensors(total_data)
    C_data_noisy = noise_injection(C_data, noise_std)
    X_coll = sample_collocation_points(NUM_COLLOCATION_POINTS).to(DEVICE)
    X_init, X_spatial = sample_boundary_points(NUM_BOUNDARY_POINTS)

    X_data, C_data_noisy = X_data.to(DEVICE), C_data_noisy.to(DEVICE)
    X_init, X_spatial = X_init.to(DEVICE), X_spatial.to(DEVICE)

    # 2. Setup Model & Trainer
    pinn_model = WoundHealingPINN(use_ffe=USE_FOURIER_FEATURES).to(DEVICE)
    loss_fn = PINNLoss(model=pinn_model, scaler=scaler, lambda_data=W_DATA, lambda_phy=W_PHY, lambda_bc=W_BC, lambda_sindy=LAMBDA_SINDY)
    trainer = PINNTrainer(model=pinn_model, loss=loss_fn)

    # 3. Train
    adam_hist = trainer.train_adam(X_data, C_data_noisy, X_coll, X_init, X_spatial, epochs=EPOCHS, lr=LR)
    lbfgs_hist, (D_f, rho_f) = trainer.train_lbfgs(X_data, C_data_noisy, X_coll, X_init, X_spatial, max_iter=LBFGS_ITER,
                                                   lr=LR)

    # 4. Quick Loss Check
    final_l, _, _, _, _, _ = loss_fn.calculate_total_loss(X_data, C_data_noisy, X_coll, X_init, X_spatial)

    return {
        'noise_std': noise_std,
        'D_final': D_f.item(),
        'rho_final': rho_f.item(),
        'Final_L_Total': final_l.item()
    }


def main():
    print("=" * 80)
    print("   PINN Wound Healing (TRAM) Project Execution")
    print("=" * 80)

    # --- PHASE 1: Data Ingestion ---
    logger.info("PHASE 1: Data Preparation and Ingestion")
    try:
        raw_cell_density_data = import_data()
    except Exception as e:
        logger.error(f"FATAL ERROR: Ingestion failed. Error: {e}")
        return

    if raw_cell_density_data.empty:
        logger.error("Ingested DataFrame is empty. Cannot proceed.")
        return

    # visualize_plots(raw_cell_density_data) # Uncomment if you want to see plots every time

    # --- PHASE 2 & 3: Initial Setup for Primary Run ---
    logger.info("Tensor Conversion and Model Initialization")
    X_data, C_data, scaler = convert_to_tensors(raw_cell_density_data)
    C_data_noisy = noise_injection(C_data, NOISE_STD)
    X_coll = sample_collocation_points(NUM_COLLOCATION_POINTS).to(DEVICE)
    X_init, X_spatial = sample_boundary_points(NUM_BOUNDARY_POINTS)

    X_data, C_data_noisy = X_data.to(DEVICE), C_data_noisy.to(DEVICE)
    X_init, X_spatial = X_init.to(DEVICE), X_spatial.to(DEVICE)

    pinn_model = WoundHealingPINN(use_ffe=USE_FOURIER_FEATURES).to(DEVICE)
    loss_fn = PINNLoss(model=pinn_model, scaler=scaler, lambda_data=W_DATA, lambda_phy=W_PHY, lambda_bc=W_BC, lambda_sindy=LAMBDA_SINDY)
    trainer = PINNTrainer(model=pinn_model, loss=loss_fn)

    # --- PHASE 4: Primary Training (Adam + L-BFGS) ---
    logger.info("Starting Primary Training Run")

    adam_loss_history = trainer.train_adam(X_data, C_data_noisy, X_coll, X_init, X_spatial, epochs=EPOCHS, lr=LR)
    lbfgs_loss_history, (D_final, rho_final) = trainer.train_lbfgs(X_data, C_data_noisy, X_coll, X_init, X_spatial,
                                                                   epochs=LBFGS_ITER, lr=LR)
    if SINDY_MODE:
        logger.info("PHASE 6: SINDy Sparse Equation Discovery Results")
        # Assuming your model/loss has a way to provide the coefficient vector
        coeffs = pinn_model.sindy_coefficients.cpu().detach().numpy().flatten()
        # Terms defined in pinn_model.py
        terms = ['1', 'C', 'C^2', 'C(1-C)', 'C_xx+C_yy', 'C*(C_xx+C_yy)']

        print("\nDiscovered Equation:")
        equation_str = "dC/dt = "
        for i, val in enumerate(coeffs):
            if abs(val) > 1e-3:  # Sparsity threshold for printing
                equation_str += f"({val:.4f})*{terms[i]} + "
        print(equation_str.rstrip(" + "))
    else:
        D, rho = pinn_model.discovered_params
        logger.info(f"Final Discovery (Fisher-KPP): D = {D.item():.6e}, rho = {rho.item():.6e}")

    # --- PHASE 5: Evaluation & Validation ---
    logger.info("PHASE 5: Generating Evaluation Metrics")

    # 1. Plots
    plot_training_convergence(adam_loss_history, lbfgs_loss_history, "primary_run_convergence")

    # 2. Solution Reconstruction
    df_predicted_solution = generate_prediction_grid(pinn_model, scaler, GRID_RESOLUTION)

    # 3. Residual Analysis
    df_predicted_solution = analyze_residual_field(pinn_model, scaler, df_predicted_solution)
    plot_spatial_residuals(df_predicted_solution)
    logger.info(f"Reconstruction Complete. Sample predictions:\n{df_predicted_solution.head()}")

    # --- PHASE 6: Noise Robustness Benchmark ---
    logger.info("PHASE 6: Running Noise Robustness Benchmark")
    results = []
    for noise in NOISE_LEVELS:
        res = run_pinn_experiment(noise_std=noise, total_data=raw_cell_density_data)
        results.append(res)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 30)
    print("NOISE ROBUSTNESS RESULTS")
    print("=" * 30)
    print(results_df[['noise_std', 'D_final', 'rho_final', 'Final_L_Total']].to_markdown(index=False))

    logger.info("Full pipeline execution complete.")


if __name__ == "__main__":
    main()