import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Churninator.data_processing.data_cleaning import DataCleaner

def main():
    parser = argparse.ArgumentParser(description='Run DataCleaner with specified arguments.')
    parser.add_argument('--path_to_data', type=str, required=True, help='Path to the data file')
    parser.add_argument('--labels_to_encode', type=str, nargs='+', required=True, help='List of labels to encode')
    parser.add_argument('--save_file_path', type=str, required=True, help='Path to save the cleaned data')
    parser.add_argument('--verbose', type=bool, default=False, help='Enable verbose output')
    parser.add_argument('--make_plots', type=bool, default=False, help='Enable plotting')

    args = parser.parse_args()

    data_cleaner = DataCleaner(
        path_to_data=args.path_to_data,
        labels_to_encode=args.labels_to_encode,
        save_file_path=args.save_file_path,
        verbose=args.verbose,
        make_plots=args.make_plots
    )

if __name__ == "__main__":
    main()
