import pandas as pd
import os
from logging_utils import setup_logger
from wound_healing_jin.data_preprocessing import import_and_preprocess_data
from wound_healing_jin.feature_engineering import compute_temporal_derivative, compute_spatial_derivatives, construct_non_linear_terms, \
    build_design_matrix
from regressors import Regressor
from validation import calculate_rmse, plot_density_evolution
from simulation import ForwardSolver, ParameterRefiner, pivot_data_for_simulation

logger = setup_logger()
ROOT_DIR = "wound_healing_jin/data/cell_density"
L_DOMAIN = 1.0
DELTA_T = 0.01
PROCESSED_DATA_FILE = "wound_healing_jin/smoothed_density_data_processed.csv"

# SINDy model parameters
SINDY_THRESHOLD = 0.01
RIDGE_ALPHA = 1e-4

CANDIDATE_TERMS = [
    'density_gaussian',         # C (Zeroth Order)
    'grad_C',                   # dC/dx (Advection/Chemotaxis)
    'laplacian_C',              # d^2C/dx^2 (Diffusion)
    'C_logistic',               # C(1 - C/K) (Growth)
    'C_pow2',
    'C_gradC'
]

if __name__ == "__main__":
    logger.info("---------------------------------------------")
    logger.info("--- Inference on  Wound Healing Pipeline Started ---")
    logger.info("---------------------------------------------")

    # --- SETUP: Ensure results directory exists ---
    if not os.path.exists("results"):
        os.makedirs("results")

    # --- PHASE 1: Data Ingestion & Preprocessing (with Caching) ---
    try:
        logger.info("\n--- Starting Phase 1: Data Preprocessing ---")

        # 1. Try to load the processed data from the CSV cache
        try:
            processed_df = pd.read_csv(PROCESSED_DATA_FILE)

            if 't_coordinate' not in processed_df.columns:
                processed_df['t_coordinate'] = processed_df['time_step_index'] * DELTA_T
            if 'source_group' not in processed_df.columns:
                if 'source_file' in processed_df.columns:
                    processed_df['source_group'] = processed_df['source_file'].apply(
                        lambda x: x.split('_')[2] if len(x.split('_')) > 2 else 'default_group')
                else:
                    logger.warning("Cannot reconstruct 'source_group' from cache. Using 'initial_cells'.")
                    processed_df['source_group'] = 'initCells' + processed_df['initial_cells'].astype(str)

            logger.info(f" Complete. Loaded {len(processed_df)} rows from cached file: {PROCESSED_DATA_FILE}.")

        except FileNotFoundError:
            logger.info(f"Cached file '{PROCESSED_DATA_FILE}' not found. Running full preprocessing...")

            # 2. If cache not found, run the full expensive function
            processed_df = import_and_preprocess_data(root_dir=ROOT_DIR, delta_t=DELTA_T)

            # 3. Save the result to the cache file for next time
            processed_df.to_csv(PROCESSED_DATA_FILE, index=False)
            logger.info(f"Full preprocessing complete. Data saved to cache: {PROCESSED_DATA_FILE}.")

    except Exception as e:
        logger.critical(f"Pipeline Stopped during preprocessing: {e}")
        exit()

    # --- PHASE 2: Feature Engineering (The "Basis" Library) ---
    logger.info("\n--- Starting Feature Engineering ---")

    try:
        # 2.1 Compute Temporal Derivative (dC/dt)
        features_df = compute_temporal_derivative(processed_df, dt=DELTA_T, density_col='density_gaussian')

        # 2.2 & 2.3 Compute Spatial Derivatives ($\nabla C$ and $\nabla^2 C$)
        features_df = compute_spatial_derivatives(features_df, density_col='density_gaussian')

        # 2.4 Construct Non-Linear Terms (Assuming K=1.0 for normalized density)
        features_df = construct_non_linear_terms(features_df, density_col='density_gaussian', carrying_capacity=1.0)

        # 2.5 Build the Design Matrix (Theta) and Target Vector (Y)
        Theta, Y, clean_indices = build_design_matrix(features_df, terms=CANDIDATE_TERMS)

        essential_cols_for_validation = [
                                           'source_group', 'x_coordinate', 't_coordinate', 'density_gaussian', 'dC_dt'
                                       ] + CANDIDATE_TERMS


        features_df_clean = features_df.loc[clean_indices, essential_cols_for_validation].reset_index(drop=True)

        Theta = Theta.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        logger.info("Feature Engineering Complete. Theta and Y constructed.")

    except KeyError as e:
        logger.critical(
            f"Pipeline Stopped during Feature Engineering (KeyError): {e}. Check CANDIDATE_TERMS or density column name.")
        exit()


    # ---  Model Discovery (Sparse Regression) ---
    logger.info("\n--- Starting  Model Discovery ---")

    try:
        model = Regressor(threshold=SINDY_THRESHOLD, alpha=RIDGE_ALPHA)

        # 3.1 & 3.2 Run STRidge for discovery and variable selection
        Xi_stridge = model.fit_stridge(Theta, Y)

        # 3.3 Equation Extraction
        discovered_pde = model.extract_equation(decimals=5)

        logger.info("\n---------------------------------------------")
        logger.info("  DISCOVERED EQUATION (STRidge):")
        logger.info(f"  {discovered_pde}")
        logger.info("---------------------------------------------")

    except Exception as e:
        logger.critical(f"Pipeline Stopped during Model Discovery: {e}")
        exit()

    required_cols = ['source_group', 'x_coordinate', 't_coordinate']
    missing_cols = [col for col in required_cols if col not in features_df.columns]

    if missing_cols:
        logger.warning(f"Metadata columns missing in features_df: {missing_cols}. Merging from processed_df.")
        merge_keys = [col for col in processed_df.columns if col in features_df.columns and col not in required_cols]

        if 'source_file' in features_df.columns and 'time_step_index' in features_df.columns:
            merge_keys = ['source_file', 'time_step_index', 'x_coordinate']

        metadata_to_add = processed_df[merge_keys + required_cols].drop_duplicates()

        features_df = features_df.merge(
            metadata_to_add,
            on=merge_keys,
            how='left',
            suffixes=('', '_processed')
        )

        # Clean up in case any key column was duplicated due to complexity
        for col in features_df.columns:
            if col.endswith('_processed'):
                features_df.drop(columns=[col], inplace=True)

        logger.info("Metadata successfully restored for Phase 4.")

    # --- PHASE 4: Validation, Refinement, & Simulation ---
    logger.info("\n--- Starting Validation & Simulation ---")

    try:
        # --- Preparation: Pivot data for solver ---
        x_grid, t_eval, C_target_all = pivot_data_for_simulation(
            features_df,
            value_col='density_gaussian'
        )

        # 4.1 Parameter Refinement (Optimization)
        refiner = ParameterRefiner(
            x_grid=x_grid,
            all_term_names=CANDIDATE_TERMS,
            C_target_all=C_target_all,
            t_eval=t_eval
        )

        # Optimize the coefficients found by STRidge
        optimized_Xi, final_rmse = refiner.refine_coefficients(
            initial_Xi=Xi_stridge,
            method='L-BFGS-B',
            options={'maxiter': 500}  # Fewer iterations for faster run
        )

        # Update the model with optimized coefficients and extract final equation
        model.coefficients = optimized_Xi
        refined_pde = model.extract_equation(decimals=5)

        logger.info(f"Refined Equation: {refined_pde}")
        logger.info(f"Final Optimized RMSE: {final_rmse:.6f}")

        # 4.2 Forward Solver (Final Simulation)
        solver = ForwardSolver(
            x_grid=x_grid,
            coeffs=optimized_Xi,
            term_names=CANDIDATE_TERMS
        )

        C0 = C_target_all[:, 0]  # Initial condition is the first time snapshot
        t_span = (t_eval[0], t_eval[-1])

        sim_result = solver.solve(C0, t_span, t_eval)
        C_sim_all = sim_result.y

        # 4.3 Validation & Visualization
        validation_rmse = calculate_rmse(C_sim_all, C_target_all)

        # Generate plot comparing simulation vs experiment
        plot_density_evolution(
            x_grid=x_grid,
            t_eval=t_eval,
            C_sim=C_sim_all,
            C_target=C_target_all,
            rmse=validation_rmse,
            equation_str=refined_pde,
            output_filename='sindy_validation_result.png'
        )

        logger.info("Validation and Simulation is COMPLETE: Validation plot generated.")

    except Exception as e:
        logger.critical(f"Pipeline Stopped during Phase 4: {e}")
        exit()

    logger.info("\n---------------------------------------------")
    logger.info("---Inference Pipeline Finished Successfully ---")
    logger.info("---------------------------------------------")





