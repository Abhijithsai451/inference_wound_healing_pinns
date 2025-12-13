import pandas as pd
import numpy as np
from logging_utils import setup_logger

logger = setup_logger()


# ----------------------------------------------------------------------
# A. Temporal Derivative (dC/dt)
# ----------------------------------------------------------------------
def compute_temporal_derivative(df: pd.DataFrame, dt: float, density_col: str = 'density_gaussian') -> pd.DataFrame:
    """
    Calculates the temporal rate of change of cell density (dC/dt) using numpy.gradient,
    preserving all metadata columns by using a grouped transform.
    """
    logger.info(f"Computing temporal derivative (dC/dt) on column {density_col} using grouped transform.")

    # Define the groups: A derivative is calculated for each unique spatial point across time,
    # and separately for each experimental group (if present).
    group_cols = [col for col in df.columns if col not in ['x_coordinate', 't_coordinate', density_col]]

    # Ensure source_group is used if it exists, otherwise rely on the base grouping columns
    if 'source_group' in df.columns:
        grouping = ['source_group', 'x_coordinate']
    else:
        # Fallback grouping to define a single time series
        grouping = ['x_coordinate']

        # 1. Sort the data by time within each group to ensure correct derivative calculation
    df = df.sort_values(by=grouping + ['t_coordinate'])

    # 2. Use groupby and transform to calculate the temporal gradient for each time series
    # transform returns a series with the same index as the input, keeping all metadata intact.
    df['dC_dt'] = df.groupby(grouping)[density_col].transform(
        lambda x: np.gradient(x.values, dt)
    )

    # 3. Handle potential edge case where duplicates led to pivot aggregation earlier.
    # We must ensure there is only one dC_dt value per (x, t) point.

    logger.info("Temporal derivative dC/dt successfully calculated and added.")
    return df


# ----------------------------------------------------------------------
# B. Spatial Derivatives ($\nabla C$ and $\nabla^2 C$)
# ----------------------------------------------------------------------
def compute_spatial_derivatives(df: pd.DataFrame, density_col: str = 'density_gaussian') -> pd.DataFrame:
    """
    Calculates the first (gradient, $\nabla C$) and second (Laplacian, $\nabla^2 C$) spatial derivatives.
    (This function uses df.groupby('t_coordinate').apply and returns a new DataFrame that is joined
    back by index, which is prone to issues. We assume the current structure is working but note the risk.)
    """
    logger.info("Computing spatial derivatives (dC/dx and d^2C/dx^2).")

    def calculate_gradient_1d(group_df: pd.DataFrame):
        # ... (Existing logic for calculating derivatives remains the same) ...

        x_coords = group_df['x_coordinate']
        dx = x_coords.diff().iloc[1]
        C_data = group_df[density_col].values

        grad_C = np.gradient(C_data, dx, edge_order=2)
        laplacian_C = np.gradient(grad_C, dx, edge_order=2)

        return pd.DataFrame({
            'grad_C': grad_C,
            'laplacian_C': laplacian_C
        }, index=group_df.index)

    # Note: If 'source_group' is present, it should be part of the grouping key here as well.
    grouping = ['t_coordinate']
    if 'source_group' in df.columns:
        grouping = ['source_group', 't_coordinate']

    new_features = df.groupby(grouping, group_keys=False).apply(calculate_gradient_1d)

    # Join the new feature columns back to the main DataFrame
    df['grad_C'] = new_features['grad_C']
    df['laplacian_C'] = new_features['laplacian_C']

    logger.info("Spatial derivatives successfully calculated.")
    return df


# ----------------------------------------------------------------------
# C. Construct Non-Linear Terms
# ----------------------------------------------------------------------

def construct_non_linear_terms(df: pd.DataFrame, density_col: str = 'density_gaussian',
                               carrying_capacity: float = 1.0) -> pd.DataFrame:
    # ... (Logic remains the same, as it only adds new columns to the existing df) ...
    logger.info("Constructing non-linear candidate features.")
    C = df[density_col]

    df['C_logistic'] = C * (1 - C / carrying_capacity)
    df['C_pow2'] = C ** 2
    df['C_pow3'] = C ** 3

    if 'grad_C' in df.columns:
        df['C_gradC'] = C * df['grad_C']
    else:
        logger.warning("Cannot construct 'C_gradC': 'grad_C' column not found (spatial gradient not yet computed).")

    if 'laplacian_C' in df.columns:
        df['C_laplacian_C'] = C * df['laplacian_C']

    logger.info(f"Non-linear terms added: C_logistic, C_pow2, C_pow3, C_gradC, and C_laplacian_C.")
    return df


# ----------------------------------------------------------------------
# D. Build the Design Matrix Theta
# ----------------------------------------------------------------------

def build_design_matrix(df: pd.DataFrame, terms: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    # ... (Logic remains the same) ...
    logger.info("Building the Design Matrix (Theta) and Target Vector (Y).")

    Y = df['dC_dt'].to_frame(name='dC_dt')
    Theta = df[terms]

    combined = pd.concat([Y, Theta], axis=1).dropna()
    cleaned_indices = combined.index
    Theta_final = combined[terms]
    Y_final = combined['dC_dt'].to_frame(name='dC_dt')

    logger.info(f"Design Matrix (Theta) built. Initial rows: {len(df)}. Final rows after cleaning: {len(Theta_final)}.")
    logger.info(f"Theta Shape: {Theta_final.shape} (Rows: Space-Time, Columns: Candidate Terms)")
    logger.info(f"Y Shape: {Y_final.shape}")

    return Theta_final, Y_final, cleaned_indices