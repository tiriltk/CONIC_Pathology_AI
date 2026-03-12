import numpy as np
import os
from PIL import Image

"""
Takes the tp_results.npy file saved during inference and makes the patches into PNG images with colored cell types. 
The patches are further used in TIA script to make WSI. 
"""

#tp_results_data = np.load("/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func043_ST_HE_40x_BF_01/tp_results/tp_results.npy")
#print(f"Shape: {tp_results_data.shape}") #Shape: (625, 2048, 2048, 1)
#print(tp_results_data.min(), tp_results_data.max()) #0.0 5.0

def colored_png(tp_results_path: str, output_dir: str, offset: int):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tp_result = np.load(tp_results_path)
    data = np.squeeze(tp_result, axis=-1) #squeeze the last dimention from (625, 2048, 2048, 1) to (625, 2048, 2048)
    num_patches = tp_result.shape[0]

    #Change color to correct dataset!
    #Same color scheme as hovernet visualization 
    #dictionary, keys: class numbers, value: RGB colors
    colors_pannuke = {0: (0, 0, 0), 1: (0, 200, 255), 2: (0, 255, 0), 3: (255, 255, 0), 4: (127, 127, 127), 5: (255, 0, 0)}
    # color order nuclei: background (black), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (red)
    colors_conic = {0: (0, 0, 0), 1: (0, 0, 0), 2: (255, 0, 0), 3: (255, 0, 255), 4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 255, 0)}
    # color order nuclei: background (black), neutrophil (black), epithelial (red), lymphocyte (magenta), plasma (dark blue), eosinophil (green), connective (yellow)

    for i in range(num_patches):
        single_patch = data[i] #single patches, patch.shape = (2048, 2048)
        patch = single_patch.astype(np.uint8) #PIL requirement
        height = patch.shape[0]
        width = patch.shape[1]
        color_patch = np.zeros((height, width, 3), dtype=np.uint8) #black empty canvas, color_patch.shape = (2048, 2048, 3)
        for cell_class, color in colors_conic.items(): #remember to change to correct dataset color!
            class_mask = (patch == cell_class)
            color_patch[class_mask] = color
        img = Image.fromarray(color_patch) #converts array into PIL image 
        img.save(os.path.join(output_dir, f"tp_patch_{i+offset}.png")) 
        #offset is necessary to have different file names on the patches because the tp results were saved in two inference rounds
        #and we combine the patches into a the same folder on the computer to make WSI with TIA script

    print(f"Saved: {output_dir}")

if __name__ == "__main__":
    # colored_png(
    #     "/Volumes/Expansion/biopsy_results/conic/40x/output_fill/Func116_ST_HE_40x_BF_01/tp_results/tp_results_from_0_to_599.npy",
    #     "/Volumes/Expansion/biopsy_results/conic/40x/output_fill/Func116_ST_HE_40x_BF_01/tp_results/tp_results_colors_part1",
    #     offset = 0
    # )

    colored_png(
         "/Volumes/Expansion/biopsy_results/conic/40x/output_fill/Func116_ST_HE_40x_BF_01/tp_results/tp_results_from_600_to_1087.npy",
         "/Volumes/Expansion/biopsy_results/conic/40x/output_fill/Func116_ST_HE_40x_BF_01/tp_results/tp_results_colors_part2",
         offset = 600
    )
