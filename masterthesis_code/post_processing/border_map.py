import cv2
import numpy as np
import os

"""
Set the border pixels to black pixels to further use on the fill type map to separate overlapping cells.
"""

#File paths
borderonly_path = "/Volumes/Expansion/biopsy_results/conic/20x/output_border_only/Func116_ST_HE_20x_BF_01/wsi_border/whole_image_scaled.png"
output_dir = "/Volumes/Expansion/biopsy_results/conic/20x/output_border_only/Func116_ST_HE_20x_BF_01/wsi_border/"
os.makedirs(output_dir, exist_ok=True)

def border_map(borderonly_path, output_path):
    black_color = np.array([0,0,0])
    white_color = np.array([255,255,255])

    border_image = cv2.imread(borderonly_path)
    gray_image = cv2.cvtColor(border_image, cv2.COLOR_BGR2GRAY) #Convert to grayscale

    threshold = 10 #Tried different values
    border_mask = gray_image > threshold #Border pixels above a threshold

    result_image = np.ones_like(border_image) * white_color  #White background
    result_image[border_mask] = black_color #Black border pixels

    cv2.imwrite(output_path, result_image) #Save
    print(f"Saved: {output_path}")

output_path = os.path.join(output_dir, "black_bordermap_test.png")
border_map(borderonly_path, output_path)
