#Apply co registration

"""
Have co-registered the biopsy, and now wants to use the same transformation on the type map.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_visium = "/Volumes/Expansion/Co-registration/Func116HEVisium.tif" #Visium
#path_type_map = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func116_ST_HE_40x_BF_01/wsi_tp_results/Func116_tpmap_scaled.png" #Type map
path_type_map = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/wsi_border/correct_result/bordered_type_map.png"
path_matrix = "/Volumes/Expansion/biopsy_results/pannuke/40x/co_registration/co_reg_biopsy/Func116_affine_transform.npy" #Affine matrix
dir_save = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_border/Func116_ST_HE_40x_BF_01/wsi_border/correct_result/co_reg_type_map2/" #Saving directory

os.makedirs(dir_save, exist_ok=True)


#Functions from image_co_registation:
#Compute scaling factor
def compute_scaling_factor(fixed_image, moving_image):  #Fixed: Visium, moving: HoverNet results
    height_fixed, width_fixed = fixed_image.shape[:2]
    height_moving, width_moving = moving_image.shape[:2]

    scaleH = height_fixed / height_moving
    scaleW = width_fixed / width_moving

    #scale_factor = np.mean([scaleH, scaleW])

    return scaleW, scaleH

#Rotation and translation
def func_manual_rotation(image, angle, tx, ty):
    #Rotating by a chosen angle
    #Get image dimensions
    height, width = image.shape[:2] #take the two first values from (H, W, C)

    #Define the rotation parameters
    center = (width // 2, height // 2) #Center of rotation

    rotation_angle = angle #Chosen rotation angle in degrees (counter-clockwise)
    scale = 1.0 #Scaling factor

    #Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)

    #Perform the affine transformation
    #rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    #Nearest neighbour
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)

    #Translation values
    translate_x = tx
    translate_y = ty
    translation_matrix = np.float32([[1, 0, translate_x],[0, 1, translate_y]])

    #translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))
    translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height), flags=cv2.INTER_NEAREST)

    return translated_image


def apply_registration(visium_image_path: str, affine_matrix_path: str, type_map_path: str):
    #Load fixed_image 
    visium = cv2.imread(visium_image_path) #bgr
    visium_rgb = cv2.cvtColor(visium, cv2.COLOR_BGR2RGB)

    #Load type_map
    type_map = cv2.imread(type_map_path) #bgr
    type_map_rgb = cv2.cvtColor(type_map, cv2.COLOR_BGR2RGB)

    scaleW, scaleH = compute_scaling_factor(visium_rgb, type_map_rgb)
    type_map_scaled = cv2.resize(type_map_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_NEAREST)

    #Moving to the same dimensions as visium
    height_f, width_f = visium_rgb.shape[:2]
    type_map_resized = cv2.resize(type_map_scaled, (width_f, height_f), interpolation=cv2.INTER_NEAREST)

    #Manual rotation and translation
    manual_rotation = [8, -100, 0]  #Func116
    angle, dx, dy = manual_rotation
    mask_rotated = func_manual_rotation(type_map_resized, angle, dx, dy)

    #Load matrix
    matrix_full = np.load(affine_matrix_path) #3x3
    matrix = matrix_full[:2, :] #2x3 til warpAffine

    #Dimensions for the output image
    output_dimensions = (width_f, height_f) #visium width and height

    #transformed_type_map = cv2.warpAffine(mask_rotated, matrix, output_dimensions)
    transformed_type_map = cv2.warpAffine(mask_rotated, matrix, output_dimensions, flags=cv2.INTER_NEAREST)

    return visium_rgb, type_map_rgb, transformed_type_map

visium_rgb, type_map_original, type_map_registered = apply_registration(path_visium, path_matrix, path_type_map)

#Plotting
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(visium_rgb)
plt.title("HE Visium")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(type_map_original)
plt.title("Original type map")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(visium_rgb)
plt.imshow(type_map_registered, alpha=0.5)
plt.title("Type map registrert")
plt.axis("off")

overlay_path = os.path.join(dir_save, "Func116_tpmap_registered_overlay.png")
plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
print("Saved overlay:", overlay_path)

plt.tight_layout()
plt.show()

save_path = os.path.join(dir_save, "Func116_tpmap_registered.png")
cv2.imwrite(save_path, cv2.cvtColor(type_map_registered, cv2.COLOR_RGB2BGR))
print("Saved:", save_path)



#Select a tight box around the circular biopsy to scale better as the biopsies have different sizes
#Instead of using the whole image with lots of background
# def biopsy_mask(rgb_image, threshold=230):
#     gray_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
#     mask = gray_img < threshold
#     ys, xs = np.where(mask)
#     y_min, y_max = ys.min(), ys.max()  #top and bottom
#     x_min, x_max = xs.min(), xs.max()  #left and right
#     return x_min, y_min, x_max, y_max

#Find the boxes, same as in image_co_registration:
    #Biopsy mask finds box around the biopsies
    # fx1, fy1, fx2, fy2 = biopsy_mask(visium_rgb) #fixed
    # mx1, my1, mx2, my2 = biopsy_mask(type_map_rgb) #moving

    # #Crops out the box with the biopsies to make registration better
    # fixed_crop  = visium_rgb[fy1:fy2, fx1:fx2] 
    # moving_crop = type_map[my1:my2, mx1:mx2]

    # #Scaling the cropped biopsy
    # height_f = fixed_crop.shape[0]
    # width_f  = fixed_crop.shape[1]

    # height_m = moving_crop.shape[0]
    # width_m  = moving_crop.shape[1]

    # scaleH = height_f / height_m
    # scaleW = width_f  / width_m

    #Scale to same biopsy size with the scaling factors
    #type_map_scaled = cv2.resize(type_map_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_NEAREST)