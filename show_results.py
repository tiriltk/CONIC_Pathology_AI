import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def read_metrics_data(file_path, encoding):
    """Read metrics from CSV file."""
    df = pd.read_csv(file_path, encoding=encoding)
    return df

def load_npy_images(npys_path):
    """Load images from a .npy file."""
    images = np.load(npys_path)
    return images  # Expect images to be in shape (num_images, height, width, channels)

def overlay_cell_types(original_images, df):
    """Overlay the cell types on the original images based on counts in the DataFrame."""
    # Define colors for different cell types
    colors = {
        "neutrophil": [255, 0, 0],  # Red
        "epithelial": [0, 255, 0],  # Green
        "lymphocyte": [0, 0, 255],  # Blue
        "plasma": [255, 255, 0],    # Cyan
        "eosinophil": [255, 0, 255], # Magenta
        "connective": [0, 255, 255]  # Yellow
    }

    # Loop through each image in the dataset
    for i, original_image in enumerate(original_images):
        original_image = original_image.astype(np.uint8)  # Ensure image is in the right type

        # Create an overlay image
        overlay = np.zeros_like(original_image)

        # Loop through each cell type and add color masks based on counts
        for cell_type, color in colors.items():
            # Get the count of nuclei for this cell type
            if cell_type in df.columns:
                count = df[cell_type].iloc[0]  # Get the count for the first image

                # Generate random positions for the current cell type (for visualization purposes)
                for _ in range(int(count)):
                    y = np.random.randint(0, original_image.shape[0])
                    x = np.random.randint(0, original_image.shape[1])

                    # Ensure the position is within the bounds of the image
                    if y < overlay.shape[0] and x < overlay.shape[1]:
                        overlay[y, x] = color  # Place the colored cell

        # Create a final image by blending original and overlay
        final_image = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)

        # Display the final image
        plt.figure(figsize=(10, 10))
        plt.imshow(final_image)
        plt.axis('off')
        plt.title(f"H&E Image with Cell Type Overlay (Image {i+1})")
        plt.show()

# Example usage
npy_file_path = 'csv_file_path = f"/media/jenny/PRIVATE_USB/AugHoverData/tester/ensemble_all_fold_0_conic/hover_paper_conic_seresnext50_00/00/results/49.csv'
csv_file_path = '/media/jenny/PRIVATE_USB/AugHoverData/data/images.npy'  # Path to original H&E image

# Read the CSV and load the images
metrics_df = read_metrics_data(csv_file_path,encoding='ISO-8859-1' )
original_images = load_npy_images(npy_file_path)

# Overlay the cell types on the images
overlay_cell_types(original_images, metrics_df)

