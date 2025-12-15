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