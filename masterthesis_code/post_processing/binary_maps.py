import numpy as np
import cv2
import os

"""
Make binary map and colored map for each cell class from the co-registered type map with border image.
"""

#File paths
typemap_path = "/Volumes/Expansion/biopsy_results/conic/20x/co_registration/Func116_ST_HE_40x_BF_01/Func116_tpmap_registered.png"
out_dir = "/Volumes/Expansion/biopsy_results/conic/20x/co_registration/"
os.makedirs(out_dir, exist_ok=True)
typemap = cv2.imread(typemap_path) #Load typemap image

#The colors used in the model:
#PanNuke color order nuclei: background (black), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey), epithelial (red)
pannuke_colors = {0: (0, 0, 0), 1: (255, 200, 0), 2: (0, 255, 0), 3: (0, 255, 255), 4: (127, 127, 127), 5: (0, 0, 255)}
#CoNIC color order nuclei: background (black), neutrophil (black), epithelial (red), lymphocyte (magenta), plasma (dark blue), eosinophil (green), connective (yellow)
conic_colors = {0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 255), 3: (255, 0, 255), 4: (255, 0, 0), 5: (0, 255, 0), 6: (0, 255, 255)}

threshold = 10 #Selected threshold value

for cell_class, color in conic_colors.items(): #Loop through the cell classes and colors
    difference = np.abs(typemap - color) #Absolute difference between type map pixels color and assigned pixel color values
    mask = np.all(difference < threshold, axis=2) # Mask (boolean) for segmenting pixels if difference is below threshold for all color channels

    binary_image = (mask.astype(np.uint8)) * 255  #Make binary map by setting masked pixels to white
    binary_path = os.path.join(out_dir, f"type{cell_class}binary.png")
    cv2.imwrite(binary_path, binary_image) #Save binary map

    color_image = np.zeros_like(typemap) #Create black canvas
    color_image[mask] = color #Apply the cell class color to the segmented pixels
    color_path = os.path.join(out_dir, f"type{cell_class}colored.png")
    cv2.imwrite(color_path, color_image) #Save colored map
