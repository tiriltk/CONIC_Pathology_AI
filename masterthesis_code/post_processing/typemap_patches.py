import numpy as np
import os
from PIL import Image

"""
Takes the tp_results.npy file saved during inference and makes the patches into PNG images with colored cell types. 
The patches are later used in TIA script to make WSI to get type map. 
"""

def typemap_patches(tp_results_path, output_dir, offset):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #PanNuke color order nuclei: background (black), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (red)
    #CoNIC color order nuclei: background (black), neutrophil (black), epithelial (red), lymphocyte (magenta), plasma (dark blue), eosinophil (green), connective (yellow)

    pannuke_colors = {0: (0, 0, 0), 1: (0, 200, 255), 2: (0, 255, 0), 3: (255, 255, 0), 4: (127, 127, 127), 5: (255, 0, 0)}
    conic_colors = {0: (0, 0, 0), 1: (0, 0, 0), 2: (255, 0, 0), 3: (255, 0, 255), 4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 255, 0)}
    
    tp_result = np.load(tp_results_path)
    data = np.squeeze(tp_result, axis=-1) #squeeze the last dimention from (625, 2048, 2048, 1) to (625, 2048, 2048)
    num_patches = data.shape[0]
    
    for i in range(num_patches):
        patch = data[i].astype(np.uint8) #patch.shape = (2048, 2048)
        height, width = patch.shape
        color_patch = np.zeros((height, width, 3), dtype=np.uint8) #black background, color_patch.shape = (2048, 2048, 3)

        for cell_class, color in conic_colors.items(): #Remember to change to correct dataset!
            class_mask = (patch == cell_class) #Binary mask
            color_patch[class_mask] = color #Apply color

        img = Image.fromarray(color_patch) #Convert array into PIL image 
        img.save(os.path.join(output_dir, f"typepatch_{i+offset}.png")) #Save 
    print(f"Saved: {output_dir}")

if __name__ == "__main__":
    typemap_patches(
        "/Volumes/Expansion/biopsy_results/conic/20x/output_border/Func116_ST_HE_20x_BF_01/Func116_ST_HE_20x_BF_01/tp_results/tp_results_from_0_to_271.npy",
        "/Volumes/Expansion/biopsy_results/conic/20x/output_border/Func116_ST_HE_20x_BF_01/Func116_ST_HE_20x_BF_01/tp_results/tp_results_colors_test",
        offset = 0
    )

    # typemap_patches(
    #      "/Volumes/Expansion/biopsy_results/conic/40x/output_fill/Func116_ST_HE_40x_BF_01/tp_results/tp_results_from_600_to_1087.npy",
    #      "/Volumes/Expansion/biopsy_results/conic/40x/output_fill/Func116_ST_HE_40x_BF_01/tp_results/tp_results_colors_part2",
    #      offset = 600
    # )

#tp_results_data = np.load("/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func043_ST_HE_40x_BF_01/tp_results/tp_results.npy")
#print(f"Shape: {tp_results_data.shape}") #Shape: (625, 2048, 2048, 1)
#print(tp_results_data.min(), tp_results_data.max()) #0.0 5.0