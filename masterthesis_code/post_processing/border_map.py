import numpy as np
import cv2
import os

"""
Set the border pixels to black pixels to make a binary map.
The binary map is later used on the type map to separate overlapping cells.
"""

#File paths
bordermap_path = "/Volumes/Expansion/biopsy_results/conic/20x/output_border_only/Func116_ST_HE_20x_BF_01/wsi_border/whole_image_scaled.png"
output_dir = "/Volumes/Expansion/biopsy_results/conic/20x/output_border_only/Func116_ST_HE_20x_BF_01/wsi_border/"
os.makedirs(output_dir, exist_ok=True)

def border_map(bordermap_path, output_path):
    border_image = cv2.imread(bordermap_path)
    gray_image = cv2.cvtColor(border_image, cv2.COLOR_BGR2GRAY) #Convert to grayscale
    #gray_output_path = os.path.join(output_dir, "gray_borderimage.png")
    #cv2.imwrite(gray_output_path, gray_image) #Save gray image

    black_color = np.array([0,0,0])
    white_color = np.array([255,255,255])

    threshold = 10 #Tried different threshold values
    border_mask = gray_image > threshold #Border pixels above a threshold
    result_image = np.ones_like(border_image) * white_color #White background
    result_image[border_mask] = black_color #Black border pixels

    cv2.imwrite(output_path, result_image) #Save
    print(f"Saved: {output_path}")

output_path = os.path.join(output_dir, "binary_bordermap.png")
border_map(bordermap_path, output_path)

