# Imports required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def main(input_path, output_path):
    # Loading the data set
    df = pd.read_csv(input_path)

    # Columns for which 0 is invalid and must be considered as missing
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace 0 with NaN
    df[cols] = df[cols].replace(0, np.NaN)

    # Imputation of missing values by the mean
    df.fillna(df.mean(), inplace=True)

    # Rename column 'Outcome' as 'HasDiabetes'
    df.rename(columns={'Outcome': 'HasDiabetes'}, inplace=True)

    # Create the output directory if necessary
    os.makedirs(output_path, exist_ok=True)

    # Saving the cleaned dataset
    df.to_csv(os.path.join(output_path, "data_diabetesPima_clean.csv"), index=False)

    # Calculating the correlation matrix
    cor_matrix = df.corr()

    # Saving the correlation matrix in CSV format
    cor_matrix.to_csv(os.path.join(output_path, "correlation_matrix.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Raw CSV file path")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    # Launching pre-treatment
    main(args.input_path, args.output_path)

