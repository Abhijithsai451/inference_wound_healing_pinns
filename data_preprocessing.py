import os
import glob
import pandas as pd

from filters import apply_rolling_mean, apply_gaussian_smoothing
from logging_utils import setup_logger

# logger
logger = setup_logger()

# Importing the data files
ROOT_DIR = "./data/cell_density"

CUSTOM_ROLLING_WINDOW = 7
CUSTOM_GAUSSIAN_SIGMA = 1.5


# Import the .dat files from the data folder and its subfolders.
def import_data(file_paths):
    all_dfs = []

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, sep='\s+', engine='python', header=None)
            df = df.rename(columns={0: 'Density_Raw'})
            relative_path = os.path.relpath(file_path, ROOT_DIR)
            path_components = relative_path.split(os.path.sep)

            folder_name = path_components[0]
            file_name = path_components[1]
            init_cells = int(folder_name.replace('initCells',''))

            time_step_str = [part for part in file_name.split('_') if part.startswith('t')][0]
            time_step = int(time_step_str.replace('t',''))

            # Adding metadata to the Dataframes
            df['source_file'] = file_name
            df['initial_cells'] = init_cells
            df['time_step'] = time_step

            all_dfs.append(df)
            logger.info(f"Successfully processed file: {file_name}")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    return all_dfs

def apply_smoothing(df: pd.DataFrame)-> pd.DataFrame:
    """
    Applies Central Rolling Mean and Gaussian Smoothing to the dataframes.
    The Smoothing is applied independantly per source_file group.
    """
    logger.info(f"Applying smoothing filters on combines data (shape: {df.shape})")

    # A. Central Rolling Mean
    df['density_rolling_mean'] = df.groupby('source_file')['Density_Raw'].transform(
        lambda x: apply_rolling_mean(x, window=CUSTOM_ROLLING_WINDOW)
    )
    logger.info(f"Central Rolling Mean (Window={CUSTOM_ROLLING_WINDOW}) applied.")

    # B. Gaussian Smoothing
    df['density_gaussian'] = df.groupby('source_file')['Density_Raw'].transform(
        lambda x: apply_gaussian_smoothing(x, sigma=CUSTOM_GAUSSIAN_SIGMA)
    )
    logger.info(f"Gaussian Smoothing (Sigma={CUSTOM_GAUSSIAN_SIGMA}) applied.")

    return df



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

    # 3. Apply Smoothing
    smoothed_df = apply_smoothing(final_df)

    # 4. Output
    output_filename = 'smoothed_density_data_processed.csv'
    smoothed_df.to_csv(output_filename, index=False)
    logger.info(f"Processed data saved to: {output_filename}")

    logger.info("--- Data Preprocessing Pipeline Finished ---")
