import os
import cv2
import numpy as np

#Co-registered filled type map with black borders overlayed
type_map_path = "/Volumes/Expansion/biopsy_results/conic/20x/co_registration/Func116_ST_HE_40x_BF_01/Func116_tpmap_registered.png"
out_dir = "/Volumes/Expansion/biopsy_results/conic/20x/co_registration/testing2/"

os.makedirs(out_dir, exist_ok=True)
image = cv2.imread(type_map_path) #BGR format

#BGR, as opencv uses bgr
colors_pannuke = {0: (0, 0, 0), 1: (255, 200, 0), 2: (0, 255, 0), 3: (0, 255, 255), 4: (127, 127, 127), 5: (0, 0, 255)}
colors_conic = {0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 255), 3: (255, 0, 255), 4: (255, 0, 0), 5: (0, 255, 0), 6: (0, 255, 255)}

#Color range
# lower_epi_conic = np.array([0, 0, 245])
# upper_epi_conic = np.array([0, 0, 255])

# mask = cv2.inRange(image, lower_epi_conic, upper_epi_conic)
# color_path = os.path.join(out_dir, f"epi_colored.png")
# cv2.imwrite(color_path, mask)

masks = {}
threshold = 10 #if pixel is within 10 in RGB

for cell_type, color in colors_conic.items():
    color_arr = np.array(color)

    #Difference 
    diff = np.abs(image - color_arr)

    #Pixels belong to class if diff is less than threshold
    mask = np.all(diff <= threshold, axis=2)

    masks[cell_type] = mask.astype(np.uint8)

    #Without colors (binary). White is cells in a class, black is background.
    binary_img = (mask.astype(np.uint8)) * 255
    binary_path = os.path.join(out_dir, f"type_{cell_type}_binary.png")
    cv2.imwrite(binary_path, binary_img)

    #With colors. Correct color of cell in a class, black is background.
    color_img = np.zeros_like(image)
    color_img[mask] = np.array(color, dtype=np.uint8)
    color_path = os.path.join(out_dir, f"type_{cell_type}_colored.png")
    cv2.imwrite(color_path, color_img)
