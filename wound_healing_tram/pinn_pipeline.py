from pathlib import Path

import pandas as pd
import torch
from torch.distributed.tensor.parallel import loss_parallel

from pinn_data_preprocessing import import_data
from logging_utils import setup_logger
from wound_healing_tram.pinn_evaluation_utils import plot_training_convergence, generate_prediction_grid
from wound_healing_tram.pinn_loss_functions import PINNLoss
from wound_healing_tram.pinn_model import WoundHealingPINN, USE_FOURIER_FEATURES, get_device
from wound_healing_tram.pinn_trainer import PINNTrainer
from wound_healing_tram.tensor_data_utils import sample_collocation_points, noise_injection, sample_boundary_points
from wound_healing_tram.visualize_data import visualize_plots
from tensor_data_utils import convert_to_tensors

logger = setup_logger()
DEVICE = get_device()
NUM_COLLOCATION_POINTS = 200
NUM_BOUNDARY_POINTS = 100
NOISE_STD = 0.01
EPOCHS = 30
LR = 1e-3
GRID_RESOLUTION = 50
def main():

    print("====================================================================================================")
    print("   PINN Wound Healing (TRAM) Project Execution")
    print("====================================================================================================")

    # --- PHASE 1: Data Ingestion (Initial Step) ---
    logger.info("\n Data Preparation and Ingestion]")

    # Execute the utility function to get the combined data
    try:
        raw_cell_density_data = import_data()
    except Exception as e:
        logger.info(f"\nFATAL ERROR: Could not complete data ingestion. Check utility file BASE_PATH. Error: {e}")
        return

    if raw_cell_density_data.empty:
        logger.info("\n*** ERROR: Ingested DataFrame is empty. Cannot proceed. ***")
        return

    logger.info("\n Ingestion Complete. Data Summary:")
    logger.info(f"Total data points ingested: {len(raw_cell_density_data)}")
    logger.info(f"Unique time slices/experiments: {raw_cell_density_data['ID'].nunique()}")
    logger.debug(raw_cell_density_data.head(100))
    logger.debug(raw_cell_density_data.columns.tolist())

    logger.info("Data Visualization of the Raw Cell Density DataFrame")
    visualize_plots(raw_cell_density_data)

    logger.info("Data Refactoring and Tensor Conversion")
    X_data, C_data, scaler = convert_to_tensors(raw_cell_density_data)

    # Collocation point Generation
    X_collocation_points = sample_collocation_points(NUM_COLLOCATION_POINTS)

    # Adding noise to the data
    C_data_noisy = noise_injection(C_data, NOISE_STD)
    logger.info("Collocation Points Generated and Noise Injected.")
    # Generating Boundary Points
    X_init, X_spatial = sample_boundary_points(NUM_BOUNDARY_POINTS)
    logger.info("Moving all the data tensors to the device for GPU Acceleration.")

    X_data = X_data.to(DEVICE)
    C_data_noisy = C_data_noisy.to(DEVICE)
    X_collocation_points = X_collocation_points.to(DEVICE)
    X_init = X_init.to(DEVICE)
    X_spatial = X_spatial.to(DEVICE)

    logger.info("Neural Network Architecture>>>>")
    pinn_model = WoundHealingPINN(use_ffe=USE_FOURIER_FEATURES)
    pinn_model = pinn_model.to(DEVICE)
    loss = PINNLoss(model= pinn_model, scaler = scaler, lambda_data = 100.0, lambda_phy = 1.0)


    L_total, L_data, L_phy, L_bc, L_ic, L_neumann = loss.calculate_total_loss(
        X_data = X_data,
        C_data = C_data_noisy,
        X_collocation=X_collocation_points,
        X_initial=X_init,
        X_spatial=X_spatial
    )

    logger.info("---Initial Loss Calculation Complete.-----")

    D_init, rho_init = pinn_model.pde_params
    logger.info(f"Total Loss: {L_total:.4f}, \nData Loss: {L_data:.4f}, \nPhysics Loss: {L_phy:.4f}, \nBC Loss: {L_bc:.4f},"
                f"\nInitial Conditions Loss: {L_ic:.4f}, \nNeumann Boundary Loss: {L_neumann:.4f}"
                f" \nDiffusivity (D): {D_init}, \nProliferation (rho): {rho_init} ")

    trainer = PINNTrainer(model= pinn_model, loss = loss)

    adam_loss_history = trainer.train_adam(
        X_data=X_data,
        C_data=C_data_noisy,
        X_collocation=X_collocation_points,
        X_initial=X_init,
        X_spatial=X_spatial,
        epochs=EPOCHS,
        lr=LR
    )
    logger.info(f"\n--- Final Parameters (After L-BFGS Refinement) ---")

    lbfgs_loss_history, (D_final, rho_final) = trainer.train_lbfgs(
        X_data=X_data,
        C_data=C_data_noisy,
        X_collocation=X_collocation_points,
        X_initial=X_init,
        X_spatial=X_spatial,
        epochs=EPOCHS,
        lr=LR
    )
    logger.info(f"\n--- Final Parameters (After L-BFGS Refinement) ---")
    logger.info(f"  Diffusion Coeff D: {D_final.cpu().item():.6e}")
    logger.info(f"  Proliferation Coeff rho: {rho_final.cpu().item():.6e}")

    logger.info(f"\n--- Evaluation and Validation of the Final Model ---")

    plot_training_convergence(adam_history=adam_loss_history,
                              lbfgs_history=lbfgs_loss_history,
                              output_filename="wound_healing_tram_convergence")

    df_predicted_solution = generate_prediction_grid(
        model=pinn_model,
        scaler=scaler,
        resolution=GRID_RESOLUTION
    )
    logger.info(f"Solution reconstruction prepared. Final DataFrame head:\n{df_predicted_solution.head()}")
    logger.info("Execution complete ")

if __name__ == "__main__":
    main()
