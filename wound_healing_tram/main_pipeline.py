from pathlib import Path

import pandas as pd
from data_preprocessing import import_data
from logging_utils import setup_logger
from wound_healing_tram.visualize_data import visualize_plots
from tensor_data_utils import convert_to_tensors
logger = setup_logger()

def main():

    print("====================================================================================================")
    print("   PINN Wound Healing (TRAM) Project Execution")
    print("====================================================================================================")

    # --- PHASE 1: Data Ingestion (Initial Step) ---
    logger.info("\n Data Preparation and Ingestion]")

    # Execute the utility function to get the combined data
    try:
        raw_cell_density_data = import_data()
    except Exception as e:
        logger.info(f"\nFATAL ERROR: Could not complete data ingestion. Check utility file BASE_PATH. Error: {e}")
        return

    if raw_cell_density_data.empty:
        logger.info("\n*** ERROR: Ingested DataFrame is empty. Cannot proceed. ***")
        return

    logger.info("\n Ingestion Complete. Data Summary:")
    logger.info(f"Total data points ingested: {len(raw_cell_density_data)}")
    logger.info(f"Unique time slices/experiments: {raw_cell_density_data['ID'].nunique()}")
    logger.debug(raw_cell_density_data.head(100))
    logger.debug(raw_cell_density_data.columns.tolist())

    logger.info("Data Visualization of the Raw Cell Density DataFrame")
    visualize_plots(raw_cell_density_data)

    logger.info("Data Refactoring and Tensor Conversion")
    X_data, C_data, scaler = convert_to_tensors(raw_cell_density_data)

    print(C_data[1:10])


    logger.info("\nExecution complete ")

if __name__ == "__main__":
    main()
