import pandas as pd
import numpy as np
from logging_utils import setup_logger

logger = setup_logger()

# ----------------------------------------------------------------------
# A. Temporal Derivative (dC/dt)
# ----------------------------------------------------------------------
def compute_temporal_derivative(df: pd.DataFrame, dt:float, density_col:str='density_gaussian') -> pd.DataFrame:
    """
    Calculates the temporal rate of change of cell density (dC/dt) using numpy.gradient on column {density_gaussian}.
    across the time axis (t_coordinates)
    """
    logger.info(f"computing temporal derivative (dC/dt) on column {density_col}")

    # Check for duplicates before pivoting
    duplicates = df.groupby(['x_coordinate', 't_coordinate'])[density_col].count()
    logger.info(f"Number of duplicates before aggregation: {len(duplicates[duplicates > 1])}")

    # 1. Aggregate duplicate x_coordinate, t_coordinate pairs by taking the mean
    df = df.groupby(['x_coordinate', 't_coordinate'])[density_col].mean().reset_index()

    # 1. Pivot the smoothed Data into a 2D matrix (Space + Time)
    C_matrix = df.pivot(
        index='x_coordinate',
        columns='t_coordinate',
        values=density_col
    ).sort_index(axis=0).sort_index(axis=1)

    # 2. Calculate the Gradient across the columns (time axis)
    dC_dt_matrix = np.gradient(C_matrix.values, dt, axis=1)

    # 3. Convert the resulting 2D matrix back to "long" format
    dC_dt_df = pd.DataFrame(
        dC_dt_matrix,
        index=C_matrix.index,
        columns=C_matrix.columns
    ).stack().rename('dC_dt')

    # 4. Merge the new 'dC_dt' column back into the original long-format DataFrame
    df = df.set_index(['x_coordinate', 't_coordinate']).join(dC_dt_df).reset_index()

    logger.info("Temporal derivative dC/dt successfully calculated and added.")
    return df

# ----------------------------------------------------------------------
# B. Spatial Derivatives ($\nabla C$ and $\nabla^2 C$)
# ----------------------------------------------------------------------
def compute_spatial_derivatives(df: pd.DataFrame, density_col: str = 'density_gaussian') -> pd.DataFrame:
    """
    Calculates the first (gradient, $\nabla C$) and second (Laplacian, $\nabla^2 C$) spatial derivatives.

    Args:
        df (pd.DataFrame): The combined and smoothed DataFrame.
        density_col (str): The column containing the smoothed density data.

    Returns:
        pd.DataFrame: The DataFrame with 'grad_C' and 'laplacian_C' columns added.
    """
    logger.info("Computing spatial derivatives (dC/dx and d^2C/dx^2).")

    def calculate_gradient_1d(group_df: pd.DataFrame):
        """Helper to calculate 1st and 2nd derivatives for a single time snapshot."""

        # 1. Determine the spatial step size (dx)
        # We rely on the x_coordinate to be evenly spaced. dx is the difference between sequential points.
        x_coords = group_df['x_coordinate']
        dx = x_coords.diff().iloc[1]
        C_data = group_df[density_col].values

        # 2. Calculate the 1st derivative ($\nabla C$)
        # edge_order=2 ensures better accuracy at the boundaries.
        grad_C = np.gradient(C_data, dx, edge_order=2)

        # 3. Calculate the 2nd derivative ($\nabla^2 C$) by applying gradient again on the first derivative
        laplacian_C = np.gradient(grad_C, dx, edge_order=2)

        # 4. Return the results as a new DataFrame matching the size of the group
        return pd.DataFrame({
            'grad_C': grad_C,
            'laplacian_C': laplacian_C
        }, index=group_df.index)

    # Group by t_coordinate and apply the derivative calculation to each snapshot independently.
    # The transform/apply operation ensures that the derivatives are calculated only
    # within the boundaries of each time slice, not across time.
    new_features = df.groupby('t_coordinate', group_keys=False).apply(calculate_gradient_1d)

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
    """
    Constructs candidate non-linear terms likely to appear in biological models.

    Args:
        df (pd.DataFrame): The DataFrame containing density and derivatives.
        density_col (str): The smoothed density column.
        carrying_capacity (float): The saturation density (K) for logistic growth.

    Returns:
        pd.DataFrame: The DataFrame with new non-linear feature columns added.
    """
    logger.info("Constructing non-linear candidate features.")
    C = df[density_col]

    # 1. Logistic Growth Term: C(1 - C/K)
    # This term represents density-dependent growth that slows as the population approaches K.
    df['C_logistic'] = C * (1 - C / carrying_capacity)

    # 2. Simple Polynomial Terms: C^2, C^3
    df['C_pow2'] = C ** 2
    df['C_pow3'] = C ** 3

    # 3. Advection/Interaction Term: C * grad(C)
    # This term is common in advection/convection equations and can represent non-linear
    # movement or self-advection.
    if 'grad_C' in df.columns:
        df['C_gradC'] = C * df['grad_C']
    else:
        logger.warning("Cannot construct 'C_gradC': 'grad_C' column not found (spatial gradient not yet computed).")

    # 4. Diffusion term interaction: C * Laplacian(C)
    if 'laplacian_C' in df.columns:
        df['C_laplacian_C'] = C * df['laplacian_C']

    logger.info(f"Non-linear terms added: C_logistic, C_pow2, C_pow3, C_gradC, and C_laplacian_C.")
    return df


# ----------------------------------------------------------------------
# D. Build the Design Matrix Theta
# ----------------------------------------------------------------------

def build_design_matrix(df: pd.DataFrame, terms: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assembles the Design Matrix (Theta) from selected columns and the target vector (Y = dC/dt).

    Args:
        df (pd.DataFrame): The DataFrame containing all computed features and the target 'dC_dt'.
        terms (list): A list of column names to include as candidate features in Theta.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (Theta, Y) - The Design Matrix and the Target Vector,
                                            with rows containing NaNs removed.
    """
    logger.info("Building the Design Matrix (Theta) and Target Vector (Y).")

    # 1. Define the Target Vector (Y = dC/dt)
    Y = df['dC_dt'].to_frame(name='dC_dt')

    # 2. Define the Design Matrix (Theta) using the specified candidate terms
    # Ensure the Density_Raw column is included in the DataFrame passed to this function.
    Theta = df[terms]

    # 3. Critical Step: Handle NaNs
    # Derivatives calculated using finite differences (np.gradient) handle boundaries well,
    # but other processing steps might introduce NaNs. We must remove any row
    # where EITHER the target (Y) OR any feature (Theta) is NaN.
    combined = pd.concat([Y, Theta], axis=1).dropna()

    # Separate the cleaned matrices
    Theta_final = combined[terms]
    Y_final = combined['dC_dt'].to_frame(name='dC_dt')

    logger.info(f"Design Matrix (Theta) built. Initial rows: {len(df)}. Final rows after cleaning: {len(Theta_final)}.")
    logger.info(f"Theta Shape: {Theta_final.shape} (Rows: Space-Time, Columns: Candidate Terms)")
    logger.info(f"Y Shape: {Y_final.shape}")

    return Theta_final, Y_final