import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def read_metrics_data(file_path, encoding):
    
    # Read metrics from CSV file
    df = pd.read_csv(file_path, encoding=encoding)
    print(df)

    return df

def read_metrics_data_all_combined(file_path, encoding):
    """Read metrics from CSV file."""
    # Adjust pandas display options to show entire DataFrame
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)        # Adjust the width of the display
    
    # Initialize an empty list to hold DataFrames
    data_frames = []

    for i in range(40, 47):
        # Construct the full file name
        print(i)
        file_name = f"{file_path}{i}.csv"
        
        # Read the CSV file
        df = pd.read_csv(file_name, encoding=encoding)
        
        # Append the DataFrame to the list
        data_frames.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Reset display options back to default if needed
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')

    # Print the combined DataFrame
    print(combined_df)
    
    return combined_df

def load_npy_images(npys_path):
    """Load images from a .npy file."""
    images = np.load(npys_path)
    return images  # Expect images to be in shape (num_images, height, width, channels)

if __name__ == "__main__":

    # Read the CSV and return dataframe
    csv_file_path = "/media/.../nuclei_counts.csv"
    read_metrics_data(csv_file_path, encoding="ISO-8859-1" )
    
    # File path to npy file and return loaded images
    # npy_file_path = "/media/.../"
    # original_images = load_npy_images(npy_file_path)
    
    # Return combined dataframe
    # csv_file_path_comb = "/media/.../AugHoverData/all_pannuke_output/eval_func/pannuke_ensemble_all/00/results/"
    # combined_data = read_metrics_data_all_combined(csv_file_path_comb, "utf-8")

