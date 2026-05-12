import numpy as np
import cv2
import os

"""
Convert the colored border pixels from inference into a binary border map. The binary map is later used on the type map to separate overlapping cells.
"""

#File paths
bordermap_path = "/Volumes/Expansion/biopsy_results/conic/20x/output_border_only/Func116_ST_HE_20x_BF_01/wsi_border/whole_image_scaled.png"
output_dir = "/Volumes/Expansion/biopsy_results/conic/20x/output_border_only/Func116_ST_HE_20x_BF_01/wsi_border/"
os.makedirs(output_dir, exist_ok=True)

def border_map(bordermap_path, output_path): #Function to make binary border map
    border_image = cv2.imread(bordermap_path) #Load border map image from inference
    gray_image = cv2.cvtColor(border_image, cv2.COLOR_BGR2GRAY) #Convert the image to grayscale for intensity threshold

    threshold = 10 #Selected threshold value
    border_mask = gray_image > threshold #Mask (boolean) for segmenting border pixels above a threshold
    result_image = np.ones_like(border_image) * 255 #Create white canvas
    result_image[border_mask] = 0 #Apply black color to the segmented pixels

    cv2.imwrite(output_path, result_image) #Save binary border map
    print(f"Saved: {output_path}")

output_path = os.path.join(output_dir, "binary_bordermap.png")
border_map(bordermap_path, output_path)

