import os
import cv2
import numpy as np

"""
Take the co-registered filled type map with black borders overlayed and make bianry maps.
"""

#File paths
typemap_path = "/Volumes/Expansion/biopsy_results/conic/20x/co_registration/Func116_ST_HE_40x_BF_01/Func116_tpmap_registered.png"
out_dir = "/Volumes/Expansion/biopsy_results/conic/20x/co_registration/testing8/"
os.makedirs(out_dir, exist_ok=True)

#PanNuke color order nuclei: background (black), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (red)
pannuke_colors = {0: (0, 0, 0), 1: (255, 200, 0), 2: (0, 255, 0), 3: (0, 255, 255), 4: (127, 127, 127), 5: (0, 0, 255)}
#CoNIC color order nuclei: background (black), neutrophil (black), epithelial (red), lymphocyte (magenta), plasma (dark blue), eosinophil (green), connective (yellow)
conic_colors = {0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 255), 3: (255, 0, 255), 4: (255, 0, 0), 5: (0, 255, 0), 6: (0, 255, 255)}

typemap = cv2.imread(typemap_path) #BGR 

for cell_type, color in conic_colors.items(): #Remember to change to correct dataset!
    threshold = 10 #Threshold
    difference = np.abs(typemap - color) #Difference between type map color and assigned color 
    mask = np.all(difference < threshold, axis=2) #Pixels belong to cell type if diff is below threshold

    binary_image = (mask.astype(np.uint8)) * 255  #Binary map
    binary_path = os.path.join(out_dir, f"type{cell_type}binary.png")
    cv2.imwrite(binary_path, binary_image)

    color_image = np.zeros_like(typemap) #Color map
    color_image[mask] = color
    color_path = os.path.join(out_dir, f"type{cell_type}colored.png")
    cv2.imwrite(color_path, color_image)

#Color range
# lower_epi_conic = np.array([0, 0, 245])
# upper_epi_conic = np.array([0, 0, 255])

# mask = cv2.inRange(image, lower_epi_conic, upper_epi_conic)
# color_path = os.path.join(out_dir, f"epi_colored.png")
# cv2.imwrite(color_path, mask)