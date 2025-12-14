import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

from logging_utils import setup_logger

# logger
logger = setup_logger()

# Importing the data files
BASE_PATH = Path("data/cell_density/density/").resolve()
FOLDER_PREFIXES = ['A','B','C','D']
FOLDER_SUFFIXES = [f"{i:02d}" for i in range(1,7)]

def import_data()-> pd.DataFrame:
    """
    Scans the directory structure, reads all the .dat files and combines the cell density data into a single DataFrame.
    """
    all_data_slices = []
    for prefix in FOLDER_PREFIXES:
        for suffix in FOLDER_SUFFIXES:
            folder_name = f"{prefix}{suffix}"
            data_folder = Path(BASE_PATH) / folder_name

            if not data_folder.exists():
                logger.info(f"Warning: Directory {data_folder} does not exist. Skipping...")
                continue

            file_paths = sorted(data_folder.glob('density_filter_3_t*.dat'))

            for file_path in file_paths:
                try:
                    t_index = int(file_path.stem.split('_t')[-1])
                    df_slice = pd.read_csv(file_path, header=None, sep=',', skipinitialspace=True)
                    C_values = df_slice.values.flatten() # (x,t) -> C(x,t)
                    unique_id = f"CellDensity_{folder_name}_{t_index}"
                    temp_df = pd.DataFrame({
                        'ID': unique_id,
                        'C_Density': C_values
                    })
                    all_data_slices.append(temp_df)
                    logger.info(f"imported {unique_id} with {len(C_values)} cell density values")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
    final_df = pd.concat(all_data_slices, ignore_index=True)
    return final_df



