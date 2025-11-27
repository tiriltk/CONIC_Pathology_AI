import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

#From border type map to black border pixels
# to further use on fill type map

border_map = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/wsi_border/whole_image_scaled.png"
output_dir = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/wsi_border/"

#Output directory exists
os.makedirs(output_dir, exist_ok=True)

#Make border pixels black and background white
def convert_border_pixels(border_map_path, output_path):
    border_img = cv2.imread(border_map_path)

    black_color = np.array([0,0,0])
    white_color = np.array([255,255,255])

    #Convert to grayscale
    gray = cv2.cvtColor(border_img, cv2.COLOR_BGR2GRAY)

    #Everything not (or almost not) black is border
    border_mask = gray > 10  #threshold

    #White background
    result = np.ones_like(border_img)*white_color

    #Border pixels to black
    result[border_mask] = black_color 

    #Save
    cv2.imwrite(output_path, result)
    print(f"Image saved to {output_path}")

    #Mask for black pixels, chechs if all channels are 0
    #background_mask = np.all(img == black_color, axis=2)

    #Mask for border (all pixels not black)
    #border_mask = np.any(border_img != black_color, axis=2)

    #img[background_mask] = white_color
    #img[border_mask] = black_color

output_path = os.path.join(output_dir, "black_borders10.png")

convert_border_pixels(border_map, output_path)
