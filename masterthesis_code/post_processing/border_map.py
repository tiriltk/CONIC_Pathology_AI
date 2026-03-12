import cv2
import numpy as np
import os

"""
Set border pixels to black pixels to further use on fill type map to separate overlapping cells
"""

border_map = "/Volumes/Expansion/biopsy_results/conic/20x/output_border_only/Func116_ST_HE_20x_BF_01/wsi_border/whole_image_scaled.png"
output_dir = "/Volumes/Expansion/biopsy_results/conic/20x/output_border_only/Func116_ST_HE_20x_BF_01/wsi_border/"
os.makedirs(output_dir, exist_ok=True)

#Make border pixels black and background white
def black_border(border_map_path, output_path):
    black_color = np.array([0,0,0])
    white_color = np.array([255,255,255])

    border_image = cv2.imread(border_map_path)
    result_image = np.ones_like(border_image) * white_color  #White background

    border_mask = np.any(border_image > 50, axis=2) #Border pixels above a threshold, tried different threshold values
    result_image[border_mask] = black_color #Border pixels to black

    cv2.imwrite(output_path, result_image) #Save
    print(f"Saved: {output_path}")

output_path = os.path.join(output_dir, "black_border_test4.png")

black_border(border_map, output_path)
