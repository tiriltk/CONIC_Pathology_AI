import cv2
import numpy as np
import matplotlib.pyplot as plt

#Extract the border map from the overlayed PNG images with HoVer-Net results

#Note:
#The border colors in the overlay are blended with the HE image, so RGB values are not exact
#Correct approach:
#Run HoVer-Net inference and get only the border without overlayed on the HE image

#Load image
img = cv2.imread("/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func116_ST_HE_40x_BF_01/wsi/whole_image_complete_scaled.png")# BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

color_class = np.array([0, 200, 255]) # yellow example

#pannuke
#colors = {0: (0, 0, 0), 1: (0, 200, 255), 2: (0, 255, 0), 3: (255, 255, 0), 4: (127, 127, 127), 5: (255, 0, 0)}
#color order nuclei: backgroun (black), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (red)

mask = np.all(img == color_class, axis=2).astype(np.uint8)
cv2.imwrite("boundary_mask.png", mask)

plt.imshow(img)
plt.show()

