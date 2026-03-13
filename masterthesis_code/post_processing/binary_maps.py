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
typemap_image = cv2.imread(typemap_path) #BGR 

#PanNuke color order nuclei: background (black), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (red)
#CoNIC color order nuclei: background (black), neutrophil (black), epithelial (red), lymphocyte (magenta), plasma (dark blue), eosinophil (green), connective (yellow)

pannuke_colors = {0: (0, 0, 0), 1: (255, 200, 0), 2: (0, 255, 0), 3: (0, 255, 255), 4: (127, 127, 127), 5: (0, 0, 255)}
conic_colors = {0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 255), 3: (255, 0, 255), 4: (255, 0, 0), 5: (0, 255, 0), 6: (0, 255, 255)}

for cell_type, color in conic_colors.items(): #Remember to change to correct dataset!
    threshold = 10 #Threshold
    color_diff = np.abs(typemap_image - color) #Difference between pixel in image and assigned color 
    mask = np.all(color_diff < threshold, axis=2) #Pixels belong to cell type if diff is below threshold

    binary_image = (mask.astype(np.uint8)) * 255  #Binary map
    binary_path = os.path.join(out_dir, f"type_{cell_type}_binary.png")
    cv2.imwrite(binary_path, binary_image)

    color_image = np.zeros_like(typemap_image) #Color map
    color_image[mask] = color
    color_path = os.path.join(out_dir, f"type_{cell_type}_colored.png")
    cv2.imwrite(color_path, color_image)

#Color range
# lower_epi_conic = np.array([0, 0, 245])
# upper_epi_conic = np.array([0, 0, 255])

# mask = cv2.inRange(image, lower_epi_conic, upper_epi_conic)
# color_path = os.path.join(out_dir, f"epi_colored.png")
# cv2.imwrite(color_path, mask)