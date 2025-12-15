import pandas as pd
import numpy as np
import torch
from visualize_data import GRID_ROWS_X, GRID_COLS_Y
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from logging_utils import setup_logger

logger = setup_logger()

# Global configuration for the tensor shape
N_X = GRID_ROWS_X
N_Y = GRID_COLS_Y
N_SPATIAL_POINTS = N_X * N_Y

def _add_spatio_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time (T) from the ID column and reconstructs spatial indices (X,Y)
    """
    # --- 1. Extract Temporal Index (T) ---
    df['T_Index'] = df['ID'].apply(lambda x: int(x.split('_')[-1]))

    # --- 2. Reconstruct Spatial Indices (X, Y) ---
    # Calculate the sequential index (0 to 3359) within each unique time slice (ID)
    df['Spatial_Index'] = df.groupby('ID').cumcount()

    # Reconstruct X_Index (row index) and Y_Index (column index)
    df['X_Index'] = df['Spatial_Index'] // N_Y
    df['Y_Index'] = df['Spatial_Index'] % N_Y

    # Drop intermediate columns and keep the necessary ones
    df_structured = df[['X_Index', 'Y_Index', 'T_Index', 'C_Density']].copy()

    return df_structured


def convert_to_tensors(raw_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    """
    Converts the raw DataFrame into structured, normalized PyTorch input and output tensors.
    """
    logger.info("\n[TENSOR CONVERSION] Starting spatio-temporal feature engineering...")
    df_structured = _add_spatio_temporal_features(raw_df)

    # Combine inputs (X, Y, T) and output (C) for normalization
    input_cols = ['X_Index', 'Y_Index', 'T_Index']
    output_cols = ['C_Density']

    data_to_scale = df_structured[input_cols + output_cols].values

    # --- 3. Normalization (MinMax Scaling to [0, 1]) ---
    logger.info("Applying MinMax scaling to X, Y, T, and C...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_to_scale)

    # --- 4. Separation and Tensor Conversion ---

    # Determine the columns for inputs and output in the scaled array
    num_input_cols = len(input_cols)

    # X_tensor is the input features [X_norm, Y_norm, T_norm]
    X_tensor = torch.tensor(data_scaled[:, :num_input_cols], dtype=torch.float32)

    # C_tensor is the output data [C_norm]
    C_tensor = torch.tensor(data_scaled[:, num_input_cols:], dtype=torch.float32)

    logger.info(f"Tensor conversion complete. Total data points (N): {X_tensor.shape[0]:,}")
    logger.info(f"X_tensor shape (Input Features): {X_tensor.shape}")
    logger.info(f"C_tensor shape (Data Values): {C_tensor.shape}")

    return X_tensor, C_tensor, scaler

def sample_collocation_points(num_points: int) -> torch.Tensor:
    """
    Generates random collocation points (x,y,t) within the normalized domain [0 ,1] x [0,1] x [0,1].
    """
    logger.info(f"Generating {num_points: } random collocation points...")
    X_collocation = torch.rand(num_points, 3, dtype=torch.float32)
    logger.info(f"Collocation points generated. Shape: {X_collocation.shape}")
    return X_collocation

def sample_boundary_points(num_of_points: int)-> torch.Tensor:
    """
    Generates the points on the normalized boundaries when the initial time t= 0 and has only the spatial dimension.
    """
    logger.info(f"[BOUNDARY SAMPLER] Generating {num_of_points} boundary points...")

    N_t = int(num_of_points * 0.2)
    N_s = int(num_of_points * 0.8)

    # Initial Condition points at t = 0
    X_init = torch.rand(N_t, 3, dtype=torch.float32)
    X_init[:, 2] = 0.0 # Setting the T_norm values to minimum 0

    # Spatial Boundary Condition points
    X_bc = torch.rand(N_s, 3, dtype=torch.float32)

    # Boundary faces (x =0, x=1, y=0, y=1)
    boundary_indices = torch.randint(0,4, (N_s,))

    X_bc[boundary_indices == 0, 0] = 0.0
    X_bc[boundary_indices == 1, 0] = 1.0
    X_bc[boundary_indices == 2, 1] = 0.0
    X_bc[boundary_indices == 3, 1] = 1.0

    logger.info(f"[BOUNDARY SAMPLER] Initial Condition points: {N_t:,} | Boundary Condition points: {N_s:,}")
    return X_init, X_bc

def noise_injection(C_tensor: torch.Tensor, sigma: float = 0.05)-> torch.Tensor:
    """
    Adds Gaussian Noise to the normalized Cell Density Tensor C(x,t).
    THe noise STD(sigma -> noise percentage) is calculated as a percentage of the data's normalized range.
    """
    logger.info(f"Adding Gaussian Noise to C_tensor with STD={sigma:.2%}...")

    noise = torch.randn_like(C_tensor) * sigma
    C_tensor_noisy = C_tensor + noise
    C_tensor_noisy = torch.clamp(C_tensor_noisy, min=0.0, max=1.0)
    logger.info(f"Noise added. Shape: {C_tensor_noisy.shape}")

    return C_tensor_noisy
