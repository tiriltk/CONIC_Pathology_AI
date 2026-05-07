import numpy as np
import cv2
import os

"""
Use the binary border map on the type map to separate the overlapping cells.
"""

#File paths
typemap_path = "/Volumes/Expansion/biopsy_results/conic/20x/output_fill/Func116_ST_HE_20x_BF_01/tp_results/whole_image_scaled.png"
bordermap_path = "/Volumes/Expansion/biopsy_results/conic/20x/output_border_only/Func116_ST_HE_20x_BF_01/wsi_border/black_border.png"
output_dir = "/Volumes/Expansion/biopsy_results/conic/20x/output_fill/Func116_ST_HE_20x_BF_01/wsi_border_typemap/"
os.makedirs(output_dir, exist_ok=True)

type_map = cv2.imread(typemap_path) #Load type map image
border_map = cv2.imread(bordermap_path) #Load border map image

border_mask = np.all(border_map == 0, axis=2) #Mask (boolean) to segment all black border pixels 
type_borders = type_map.copy() #Make a copy of the type map
type_borders[border_mask] = 0 #Apply border mask on type map and set border pixels to black 

output_path = os.path.join(output_dir, "bordered_typemap.png")
cv2.imwrite(output_path, type_borders) #Save bordered type map
print(f"Saved: {output_path}")