import os
import glob
import pandas as pd
import numpy as np

from filters import apply_rolling_mean, apply_gaussian_smoothing
from logging_utils import setup_logger

# logger
logger = setup_logger()

# Importing the data files
ROOT_DIR = "./data/cell_density"

CUSTOM_ROLLING_WINDOW = 7
CUSTOM_GAUSSIAN_SIGMA = 1.5

# --- SIMULATION AND DOMAIN PARAMETERS ---
L_DOMAIN = 1.0          # Total length of the spatial domain (e.g., 1.0 unit)
NUM_SPATIAL_POINTS = 128 # The number of data points in EACH .dat file (must match data size)
DELTA_T = 0.01          # Time step size (dt) used in the simulation

#%% Data Preprocessing Functions

def define_grid_coordinates(data_length:int)-> np.ndarray:
    """
    Defines the spatial grid points (x-coordinates)based on the total domain lenth and the number of data points.
    Args:
        data_length: The number of density points in a single profile.
    return:
        np.ndarray:
    """
    dx = L_DOMAIN / data_length
    x_coords = np.linspace(0.0, L_DOMAIN, num=data_length, endpoint=False)
    logger.info(f"Grid points defined with dx={dx} and length={len(x_coords)}")

    return x_coords
# Import the .dat files from the data folder and its subfolders.
def import_and_preprocess_data(root_dir:str, delta_t:float)-> pd.DataFrame:
    """
        Handles file discovery, import, metadata extraction, and time coordinate assignment.
        """
    logger.info("Phase 1: Starting data import and preprocessing.")
    file_paths = glob.glob(os.path.join(root_dir, '**/*.dat'), recursive=True)
    logger.info(f"Found {len(file_paths)} data files.")

    all_dfs = []
    for file_path in file_paths:
        try:
            # Data Read
            df = pd.read_csv(file_path, sep='\s+', engine='python', header=None).rename(columns={0: 'Density_Raw'})

            # Metadata Extraction
            relative_path = os.path.relpath(file_path, root_dir)
            path_components = relative_path.split(os.path.sep)
            folder_name = path_components[0]
            file_name = path_components[-1]
            init_cells = int(folder_name.replace('initCells', ''))
            time_step_str = [part for part in file_name.split('_') if part.startswith('t')][0]
            time_step = int(time_step_str.replace('t', ''))

            # Metadata Assignment & Time Coordinate (t)
            df['source_file'] = file_name
            df['initial_cells'] = init_cells
            df['time_step_index'] = time_step
            df['t_coordinate'] = time_step * delta_t  # Define t-coordinate

            all_dfs.append(df)
            logger.debug(f"Processed file: {file_name}")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    if not all_dfs:
        logger.error("No DataFrames were successfully imported.")
        raise RuntimeError("No data imported.")

    final_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Data combined (Shape: {final_df.shape}). Now adding X-coordinates.")

    # 3. Define and Assign Spatial Coordinates (X-Grid)
    data_length_per_file = len(all_dfs[0])
    x_grid = define_grid_coordinates(data_length_per_file, L_DOMAIN)

    # Assign the x-coordinate to every data point using numpy tile
    final_df['x_coordinate'] = np.tile(x_grid, len(final_df) // len(x_grid))

    # 4. Apply Smoothing
    logger.info("Applying smoothing filters...")
    final_df['density_rolling_mean'] = final_df.groupby('source_file')['Density_Raw'].transform(
        lambda x: apply_rolling_mean(x, window=CUSTOM_ROLLING_WINDOW)
    )
    final_df['density_gaussian'] = final_df.groupby('source_file')['Density_Raw'].transform(
        lambda x: apply_gaussian_smoothing(x, sigma=CUSTOM_GAUSSIAN_SIGMA)
    )

    logger.info("Phase 1: Preprocessing complete. Data is smoothed and gridded.")
    return final_df





if __name__ == "__main__":
    logger.info("Data preprocessing started")

    #1. Find all files
    file_paths = glob.glob(os.path.join(ROOT_DIR, '**/*.dat'), recursive=True)
    logger.info(f"Found {len(file_paths)} data files.")

    if not file_paths:
        logger.warning(f"No files found in {ROOT_DIR}. Exiting.")
        exit()

    # 2. Import Data and Combine
    list_of_dfs = import_data(file_paths)

    if not list_of_dfs:
        logger.error("No DataFrames were successfully imported. Check logs for errors.")
        exit()

    final_df = pd.concat(list_of_dfs, ignore_index=True)
    logger.info(f"All data successfully combined. Final shape: {final_df.shape}")

    # 3 Define and Assign Spatial Coordinates
    data_length_per_file = len(list_of_dfs[0])
    if data_length_per_file != NUM_SPATIAL_POINTS:
        logger.warning(f"Expected {NUM_SPATIAL_POINTS} points but found {data_length_per_file}. Using found value.")
        NUM_SPATIAL_POINTS = data_length_per_file

    x_grid = define_grid_coordinates(NUM_SPATIAL_POINTS)

    # Now, assign the x-coordinate to every data point using the index modulo the size of the grid
    final_df['x_coordinate'] = np.tile(x_grid, len(final_df) // len(x_grid))
    logger.info("Spatial (X) coordinates successfully added to the DataFrame.")
    # 4. Apply Smoothing
    smoothed_df = apply_smoothing(final_df)

    # 4. Output
    output_filename = 'smoothed_density_data_processed.csv'
    smoothed_df.to_csv(output_filename, index=False)
    logger.info(f"Processed data saved to: {output_filename}")

    logger.info("--- Data Preprocessing Pipeline Finished ---")
