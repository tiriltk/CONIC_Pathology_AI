import numpy as np
import os
from PIL import Image

"""
Takes the tp_results.npy file saved during model inference and colors the cell classes and saves as image.
The saved patches are used in TIA script to make WSI to get the whole type map. 
"""

#Function to color the patches with cell class color and save the patches
def typemap_patches(tp_results_path, output_dir, offset):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) #Create output directory

    #The colors used in the model:
    #PanNuke color order nuclei: background (black), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey), epithelial (red)
    pannuke_colors = {0: (0, 0, 0), 1: (0, 200, 255), 2: (0, 255, 0), 3: (255, 255, 0), 4: (127, 127, 127), 5: (255, 0, 0)}
    #CoNIC color order nuclei: background (black), neutrophil (black), epithelial (red), lymphocyte (magenta), plasma (dark blue), eosinophil (green), connective (yellow)
    conic_colors = {0: (0, 0, 0), 1: (0, 0, 0), 2: (255, 0, 0), 3: (255, 0, 255), 4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 255, 0)}
    
    tp_result = np.load(tp_results_path) #Load tp results from inference with predicted cell classes
    data = np.squeeze(tp_result, axis=-1) #Squeeze the last dimension to remove singleton dimension
    num_patches = data.shape[0] #Number of patches in the tp results file

    for i in range(num_patches): #Loop over all the patches
        patch = data[i].astype(np.uint8) #Extract one patch
        height, width = patch.shape #Height and width of the patch, patch.shape = (2048, 2048)
        color_patch = np.zeros((height, width, 3), dtype=np.uint8) #Create black canvas with same size as patch

        for cell_class, color in pannuke_colors.items(): #Loop through the cell classes and colors
            class_mask = (patch == cell_class) #Mask (boolean) for the cell class. True if the pixels belong to that cell class and False otherwise.
            color_patch[class_mask] = color #Apply the cell class color to the segmented pixels

        img = Image.fromarray(color_patch) #Convert array to image 
        img.save(os.path.join(output_dir, f"typepatch_{i+offset}.png")) #Save the colored patches
    print(f"Saved: {output_dir}")

    #print("Min:", data.min()) #0.0
    #print("Max:", data.max()) #5.0

if __name__ == "__main__":
    typemap_patches(
         "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/output_fill/Func116_ST_HE_40x_BF_01/tp_results/tp_results_npy/tp_results_from_0_to_599.npy",
         "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/output_fill/Func116_ST_HE_40x_BF_01/tp_results/tp_results_colors_part1",
         offset = 0)

