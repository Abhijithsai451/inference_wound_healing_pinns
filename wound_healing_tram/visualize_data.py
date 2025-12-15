import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from wound_healing_tram.logging_utils import setup_logger

GRID_ROWS_X = 42
GRID_COLS_Y = 80
CENTER_X_INDEX = GRID_ROWS_X // 2  # Center is 21
logger = setup_logger()

def _restructure_for_spatial_plot(df: pd.DataFrame, time_slice_id: str, n_x: int = GRID_ROWS_X,
                                  n_y: int = GRID_COLS_Y) -> pd.DataFrame:
    """
    Internal helper to add X/Y indices (as integers) to the flattened data for a single time slice.
    """
    df_slice = df[df['ID'] == time_slice_id].copy().reset_index(drop=True)

    expected_points = n_x * n_y
    if len(df_slice) != expected_points:
        print(
            f"Plotting Error: Slice {time_slice_id} has {len(df_slice)} points, but expected {expected_points}. Skipping plot.")
        return pd.DataFrame()

    # Create the X and Y indices corresponding to the 2D grid structure (0 to 41, 0 to 79)
    x_indices = np.repeat(np.arange(n_x), n_y)
    y_indices = np.tile(np.arange(n_y), n_x)

    df_slice['X_Index'] = x_indices
    df_slice['Y_Index'] = y_indices

    return df_slice[['C_Density', 'X_Index', 'Y_Index']]

def plot_spatial_heatmap(df: pd.DataFrame, time_slice_id: str, save_path: str = 'output/cell_density_heatmap.png') -> None:
    """
    Generates and saves a 2D Heatmap plot of the cell density C(x,y) for a single time slice.
    """
    df_plot = _restructure_for_spatial_plot(df, time_slice_id)
    if df_plot.empty: return

    C_matrix = df_plot['C_Density'].values.reshape(GRID_ROWS_X, GRID_COLS_Y)
    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(C_matrix, aspect='auto', cmap='viridis', origin='upper')

    ax.set_title(f'Cell Density Heatmap: {time_slice_id} (Initial Wound Condition)')
    ax.set_xlabel('Spatial Index (X)')
    ax.set_ylabel('Spatial Index (Y)')

    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label('Cell Density (C)')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_density_profile(df: pd.DataFrame, time_slice_id: str, cross_section_x: int = CENTER_X_INDEX,
                         save_path: str = 'output/density_profile_plot.png'):
    """
    Generates and saves a 1D density profile plot (C vs Y-index) for a specific X-index cross-section.
    Filtering is now done robustly using integer indices.
    """
    df_plot = _restructure_for_spatial_plot(df, time_slice_id)
    if df_plot.empty: return

    # FIX: Filter using integer comparison (X_Index is now an integer)
    df_profile = df_plot[df_plot['X_Index'] == cross_section_x].copy().sort_values(by='Y_Index')

    if df_profile.empty:
        print(f"Plotting Error: Cross-section X={cross_section_x} not found.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df_profile['Y_Index'], df_profile['C_Density'], marker='o', linestyle='-', markersize=2, linewidth=1)

    ax.set_title(f'Density Profile along X-Index {cross_section_x} for {time_slice_id}')
    ax.set_xlabel('Spatial Index (Y-Axis)')
    ax.set_ylabel('Cell Density (C)')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_density_histogram(df: pd.DataFrame, save_path: str = 'output/density_histogram.png'):
    """Generates a histogram of the global distribution of all cell density values."""

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram of the C_Density column
    ax.hist(df['C_Density'].dropna(), bins=50, color='skyblue', edgecolor='black')

    ax.set_title('Global Distribution of Cell Density Values (All Slices)')
    ax.set_xlabel('Cell Density Bins')
    ax.set_ylabel('Frequency (Number of Grid Points)')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def visualize_plots(df: pd.DataFrame):
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    if output_dir.exists():
        logger.info(f"Output directory {output_dir} already exists. ")
    else:
        logger.info(f"Error: Failed to create output directory 'output/'. Please create it manually.")
    first_time_slice_id = df['ID'].iloc[0]
    logger.info(f"Visualizing initial conditions for time slice: {first_time_slice_id} ")

    center_x_index = 21
    # 2. Plot Spatial Heatmap (2D View of the Wound)
    logger.info("-> Generating 2D Spatial Heatmap (output/cell_density_heatmap.html)...")
    plot_spatial_heatmap(df, first_time_slice_id)

    # 3. Plot Density Profile (1D Cross-section)
    logger.info(f"-> Generating 1D Density Profile (X-Index {center_x_index}) (output/density_profile_plot.html)...")
    plot_density_profile(df, first_time_slice_id, center_x_index)

    # 4. Plot Global Density Histogram (all data points)
    logger.info("-> Generating Global Density Histogram (output/density_histogram.html)...")
    plot_density_histogram(df)

    logger.info("\nVisualizations complete. Check the 'output/' directory.")