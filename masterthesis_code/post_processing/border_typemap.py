import os
import cv2
import numpy as np

"""
Set the black border map on the filled type map to separate the cells
"""

#File paths
typemap_path = "/Volumes/Expansion/biopsy_results/conic/20x/output_fill/Func116_ST_HE_20x_BF_01/tp_results/whole_image_scaled.png"
bordermap_path = "/Volumes/Expansion/biopsy_results/conic/20x/output_border_only/Func116_ST_HE_20x_BF_01/wsi_border/black_border.png"
output_dir = "/Volumes/Expansion/biopsy_results/conic/20x/output_fill/Func116_ST_HE_20x_BF_01/wsi_border_typemap/"
os.makedirs(output_dir, exist_ok=True)

type_map = cv2.imread(typemap_path) #Load type map image
border_map = cv2.imread(bordermap_path) #Load border map image

border_mask = np.all(border_map == 0, axis=2) #Mask for border (black) pixels 
black_color = np.array([0,0,0])

type_borders = type_map.copy()
type_borders[border_mask] = black_color #Apply border mask on filled type map and set border pixels to black

output_path = os.path.join(output_dir, "bordered_typemap.png")
cv2.imwrite(output_path, type_borders)
print(f"Saved: {output_path}")