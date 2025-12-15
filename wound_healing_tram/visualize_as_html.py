from pathlib import Path

import pandas as pd
import numpy as np
import altair as alt
from logging_utils import setup_logger

logger = setup_logger()

GRID_ROWS_X = 42
GRID_COLS_Y = 80

def restructure_spatial_plot(df: pd.DataFrame, time_slice_id,n_x: int = GRID_ROWS_X, n_y: int = GRID_COLS_Y )-> pd.DataFrame:
    """
    Internal Helper to add X/Y indices to the flattened data for a single time slice.
    """
    df_slice = df[df['ID'] == time_slice_id].copy().reset_index(drop=True)

    if len(df_slice) != n_x * n_y:
        logger.info(f"Plotting Error: Slice {time_slice_id} has {len(df_slice)} points, but expected {n_x * n_y}. Skipping...")
        return pd.DataFrame()

    x_indices = np.repeat(np.arange(n_x), n_y)
    y_indices = np.tile(np.arange(n_y), n_x)

    df_slice['X_Index'] = x_indices
    df_slice['Y_Index'] = y_indices

    df_slice['X_Index'] = df_slice['X_Index'].astype(str)
    df_slice['Y_Index'] = df_slice['Y_Index'].astype(str)
    return df_slice[ ['C_Density','X_Index', 'Y_Index']]

def plot_spatial_heatmap(df: pd.DataFrame, time_slice_id: str, save_path: str = 'output/cell_density_heatmap.html') -> alt.Chart:
    """
    Generates the Heat map (2D) of the cell density C(x,y) for a single time slice.
    """
    df_plot = restructure_spatial_plot(df, time_slice_id)
    if df_plot.empty: return

    heatmap = alt.Chart(df_plot).mark_rect().encode(
        x=alt.X('Y_Index:O', title='Spatial Index (Y)'),
        y=alt.Y('X_Index:O', title='Spatial Index (X)', sort='descending'),
        color=alt.Color('C_Density:Q', title='Cell Density (C)', scale=alt.Scale(range='heatmap')),
        tooltip=['X_Index', 'Y_Index', 'C_Density']
    ).properties(
        title=f'Cell Density Heatmap: {time_slice_id} (Initial Wound Condition)'
    )

    # Save chart (Requires altair and possibly vegafusion)
    try:
        # Note: You may need to create the 'output' directory first: Path('output').mkdir(exist_ok=True)
        heatmap.save(save_path)
    except Exception as e:
        (f"Warning: Could not save Altair plot to HTML: {e}")

    return heatmap


def plot_density_profile(df: pd.DataFrame, time_slice_id: str, cross_section_x: int,
                         save_path: str = 'output/density_profile_plot.html') -> alt.Chart:
    """
    Generates and saves a 1D density profile plot (C vs Y-index) for a specific X-index cross-section.
    """
    df_plot = restructure_spatial_plot(df, time_slice_id)
    if df_plot.empty: return

    df_profile = df_plot[df_plot['X_Index'] == cross_section_x].copy()

    if df_profile.empty:
        logger.info(f"Plotting Error: Cross-section X={cross_section_x} not found.")
        return

    profile_plot = alt.Chart(df_profile).mark_line(point=True).encode(
        x=alt.X('Y_Index:Q', title='Spatial Index (Y-Axis)'),
        y=alt.Y('C_Density:Q', title='Cell Density (C)'),
        tooltip=['Y_Index', 'C_Density']
    ).properties(
        title=f'Density Profile along X-Index {cross_section_x} for {time_slice_id}'
    )

    try:
        profile_plot.save(save_path)
    except Exception as e:
        logger.info(f"Warning: Could not save Altair plot to HTML: {e}")

    return profile_plot


def plot_density_histogram(df: pd.DataFrame, save_path: str = 'output/density_histogram.html') -> alt.Chart:
    """Generates a histogram of the global distribution of all cell density values."""

    histogram = alt.Chart(df).mark_bar().encode(
        x=alt.X('C_Density:Q', bin=alt.Bin(maxbins=50), title='Cell Density Bins'),
        y=alt.Y('count()', title='Frequency (Number of Grid Points)'),
        tooltip=['C_Density', 'count()']
    ).properties(
        title='Global Distribution of Cell Density Values (All Slices)'
    )

    try:
        histogram.save(save_path)
    except:
        pass  # Silence secondary save warnings

    return histogram

def visualize_in_html(df: pd.DataFrame):
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



