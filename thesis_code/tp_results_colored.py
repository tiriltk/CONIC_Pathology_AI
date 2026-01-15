import numpy as np
import os
from PIL import Image

"""
Takes the tp_results.npy file and makes it into a PNG, with colored classes for cell types. 
PNG further used to make WSI so we get the type maps. 
"""

def colored_png(tp_results_path: str, output_dir: str, offset: int):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = np.load(tp_results_path)
    mask = np.squeeze(data, axis=-1) #squeeze the last dimention. From (625, 2048, 2048, 1) to (625, 2048, 2048).
    num_patches = mask.shape[0]

    #Change color to dataset used!
    #dictionary, keys: class numbers, value: RGB colors
    colors_pannuke = {0: (0, 0, 0), 1: (0, 200, 255), 2: (0, 255, 0), 3: (255, 255, 0), 4: (127, 127, 127), 5: (255, 0, 0)}
    # color order nuclei: background (black), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (red)

    colors_conic = {0: (0, 0, 0), 1:(0, 0, 0), 2: (255, 0, 0), 3: (255, 0, 255), 4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 255, 0)}
    # color order nuclei: neutrophil (black), epithelial (red), lymphocyte (magenta), plasma (dark blue), eosinophil (green), connective (yellow)

    for i in range(num_patches):
        patch = mask[i] #single patches, patch.shape = (2048, 2048)
        patch_uint8 = patch.astype(np.uint8) #pil
        height = patch_uint8.shape[0]
        weight = patch_uint8.shape[1]
        color_patch = np.zeros((height, weight, 3), dtype=np.uint8) #black empty canvas, color_patch.shape = (2048, 2048, 3)
        for label, color in colors_conic.items(): #change to dataset color
            class_mask = (patch_uint8 == label)
            color_patch[class_mask] = color
        img = Image.fromarray(color_patch)
        img.save(os.path.join(output_dir, f"tp_patch_{i+offset}.png"))

    print(f"Saved color mask {data.shape[0]} to {output_dir}")

if __name__ == "__main__":
    # colored_png("/Volumes/Expansion/biopsy_results/conic/20x/output_fill/Func116_ST_HE_20x_BF_01/tp_results/tp_results_from_0_to_271.npy",
    #              "/Volumes/Expansion/biopsy_results/conic/20x/output_fill/Func116_ST_HE_20x_BF_01/tp_results/tp_results_colors_part1",
    #             offset = 0)
    
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

    # colored_png(
    #     "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/tp_results/tp_results_from_0_to_599.npy",
    #     "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/tp_results_colors_part1",
    #     offset = 0
    # )

    # colored_png(
    #     "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/tp_results/tp_results_from_600_to_1087.npy",
    #     "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/tp_results_colors_part2",
    #     offset = 600
    # )
