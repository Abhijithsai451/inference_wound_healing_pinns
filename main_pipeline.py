import pandas as pd
import os
from logging_utils import setup_logger
from data_preprocessing import import_and_preprocess_data, DELTA_T, L_DOMAIN
from feature_engineering import compute_temporal_derivative, compute_spatial_derivatives, construct_non_linear_terms, \
    build_design_matrix
from regressors import Regressor

ROOT_DIR = "./data/cell_density"
CANDIDATE_TERMS = [
    'density_gaussian', # C (Zeroth Order)
    'grad_C',      # dC/dx (Advection/Chemotaxis)
    'laplacian_C', # d^2C/dx^2 (Diffusion)
    'C_logistic',  # C(1 - C/K) (Growth)
    'C_pow2',
    'C_gradC'
]

if __name__ == "__main__":
    logger = setup_logger(name='main_pipeline', log_file= 'logs/run_logs.log')
    logger.info("---------------------------------------------")
    logger.info("---Inference on Wound Healing Pipeline Started ---")
    logger.info("---------------------------------------------")

    # Data ingestion and Preprocessing
    try:
        processed_df = import_and_preprocess_data(root_dir=ROOT_DIR, delta_t=DELTA_T)
        logger.info("Data Import and Preprocessing Complete.")
    except Exception as e:
        logger.critical(f"Pipeline Stopped: {e}")
        exit()

    logger.info("\n --- Starting Feature Engineering ---")

    # Computing Temporal Derivative of Cell Density
    features_df  = compute_temporal_derivative(processed_df, dt = DELTA_T)

     # Computing Spatial Derivatives
    features_df = compute_spatial_derivatives(features_df)

    # Computing the non linear terms
    features_df = construct_non_linear_terms(features_df, carrying_capacity=1.0)

    Theta, Y = build_design_matrix(features_df, terms = CANDIDATE_TERMS)

    if not os.path.exists('results'):
        os.makedirs('results')

    Theta.to_csv("results/design_matrix_theta.csv", index=False)
    Y.to_csv("results/target_vector_Y.csv", index=False)
    logger.info("Feature Engineering Complete.")

    logger.info("n-- Starting the Model Discovery Phase ---")

    # Instantiate the model with chosen parameters
    # The threshold and alpha values are critical and usually tuned (hyperparameters)
    model = Regressor(threshold=0.01, alpha=1e-4)

    # --- Run STRidge for discovery ---
    Xi_stridge = model.fit_stridge(Theta, Y)

    # 3.3 Equation Extraction
    discovered_pde = model.extract_equation(decimals=5)

    logger.info("\n---------------------------------------------")
    logger.info("  DISCOVERED EQUATION:")
    logger.info(f"  {discovered_pde}")
    logger.info("---------------------------------------------")

    # Optional: Run Least Squares for comparison
    model.fit_least_squares(Theta, Y)
    ls_pde = model.extract_equation(decimals=5)
    logger.info(f"Least Squares Equation (dense): {ls_pde}")

    logger.info("Model discovered and equation extracted.")







