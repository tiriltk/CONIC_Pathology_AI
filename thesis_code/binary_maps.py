#Binary maps
import os
import cv2
import numpy as np

#Co-registered type map with borders overlayed
type_map_path = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/wsi_border/correct_result/co_reg_type_map_nearest/Func116_tpmap_registered.png"
out_dir = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/wsi_border/correct_result/binary_maps_coreg_combined_test/"

os.makedirs(out_dir, exist_ok=True)

image = cv2.imread(type_map_path) #BGR format

#colors = {0: (0, 0, 0), 1: (0, 200, 255), 2: (0, 255, 0), 3: (255, 255, 0), 4: (127, 127, 127), 5: (255, 0, 0)}
#color order nuclei: backgroun (black), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (red)

#BGR, as opencv uses bgr
colors = {0: (0, 0, 0), 1: (255, 200, 0), 2: (0, 255, 0), 3: (0, 255, 255), 4: (127, 127, 127), 5: (0, 0, 255)}

masks = {}

tolerance = 10 #if pixel is within 10 in RGB

for type, bgr in colors.items():
    color_arr = np.array(bgr)

    #Difference 
    diff = np.abs(image - color_arr)

    #Pixels belong to class if diff is less than tolerance
    mask = np.all(diff <= tolerance, axis=2)

    masks[type] = mask.astype(np.uint8)

    #Without colors (binary)
    #White means cells in a class, black is background
    binary_img = (mask.astype(np.uint8)) * 255
    binary_path = os.path.join(out_dir, f"type_class_{type}_tol{tolerance}_binary.png")
    cv2.imwrite(binary_path, binary_img)

    #With colors
    #Correct color of cell in a class, black is background
    color_img = np.zeros_like(image)
    color_img[mask] = np.array(bgr, dtype=np.uint8)

    color_path = os.path.join(out_dir, f"type_class_{type}_tol{tolerance}_colored.png")
    cv2.imwrite(color_path, color_img)
