import os
import cv2
import numpy as np

#Set the black borders on the filled type map

type_map_path = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func116_ST_HE_40x_BF_01/wsi_tp_results/Func116_tpmap_scaled.png"
black_border_path = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/wsi_border/black_borders10.png"
output_dir = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/wsi_border/"

os.makedirs(output_dir, exist_ok=True)

#Load type map
type_img = cv2.imread(type_map_path)
#Load border map
border_img = cv2.imread(black_border_path)

#Mask for border pixels
border_mask = np.all(border_img == 0, axis=2)

black_color = np.array([0,0,0])

#Apply border mask on filled type map
type_borders = type_img.copy()
type_borders[border_mask] = black_color #set border pixels to black

output_path = os.path.join(output_dir, "bordered_type_map.png")

cv2.imwrite(output_path, type_borders)
print(f"Image saved to: {output_path}")