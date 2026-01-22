import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
This script:
Have co-registered the biopsy, and now wants to use the same transformation on the type map.
"""

path_visium = "/Volumes/Expansion/Co-registration/Func116HEVisium.tif" #Visium
#path_type_map = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func116_ST_HE_40x_BF_01/wsi_tp_results/Func116_tpmap_scaled.png" #Type map
path_type_map = "/Volumes/Expansion/biopsy_results/conic/20x/output_fill/Func116_ST_HE_20x_BF_01/wsi_border_type_map/bordered_type_map.png"
path_matrix = "/Volumes/Expansion/biopsy_results/pannuke/40x/co_registration/co_reg_biopsy/Func116_affine_transform.npy" #Affine matrix
dir_save = "/Volumes/Expansion/biopsy_results/conic/20x/co_registration/Func116_ST_HE_40x_BF_01/" #Saving directory

os.makedirs(dir_save, exist_ok=True)

#Functions from image_co_registation.py:

#Compute scale factor
#Fixed: Visium
#Moving: HoVer-Net results
def compute_scale_factor(fixed_image, moving_image):  
    height_fixed, width_fixed = fixed_image.shape[:2]
    height_moving, width_moving = moving_image.shape[:2]

    scaleH = height_fixed / height_moving
    scaleW = width_fixed / width_moving

    return scaleW, scaleH


#Rotation and translation
#Rotating by a chosen angle
def manual_rotation_translation(image, angle, tx, ty):
    #Get image dimensions
    height, width = image.shape[:2] #take the two first values from (H, W, C)

    #Rotation
    #Define rotation parameters
    center = (width // 2, height // 2) #Center coordinates of image for rotation

    rotation_angle = angle #Chosen rotation angle in degrees (positive angle is counter-clockwise)
    scale = 1.0 #Scaling factor, using 1.0 to keep same size here

    #Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)
    #Apply rotation
    #warpAffine is the function to apply the transformation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    #Translation
    #Translation values
    translate_x = tx
    translate_y = ty
    translation_matrix = np.float32([[1, 0, translate_x],[0, 1, translate_y]])

    #Apply translation 
    rotated_translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))

    return rotated_translated_image


def apply_registration(visium_image_path: str, affine_matrix_path: str, type_map_path: str):

    #Load fixed_image 
    visium = cv2.imread(visium_image_path) #bgr
    visium_rgb = cv2.cvtColor(visium, cv2.COLOR_BGR2RGB)

    #Load type_map
    type_map = cv2.imread(type_map_path) #bgr
    type_map_rgb = cv2.cvtColor(type_map, cv2.COLOR_BGR2RGB)

    #Scale
    scaleW, scaleH = compute_scale_factor(visium_rgb, type_map_rgb)
    type_map_resized = cv2.resize(type_map_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_NEAREST)

    #Manual rotation and translation
    #Manual parameters [theta, dx, dy]
    #Func116 Manual parameters = [8, -100, 0]
    manual_angle = 8 #Rotate 8 degrees
    manual_dx = -100 #Move -100 pixels in x
    manual_dy = 0 #Move 0 pixels in y
    mask_rotated = manual_rotation_translation(type_map_resized, manual_angle, manual_dx, manual_dy)

    #Load matrix
    matrix_full = np.load(affine_matrix_path) #3x3
    matrix = matrix_full[:2, :] #2x3 required by warpAffine

    #Dimensions for the output image
    height_f, width_f = visium_rgb.shape[:2]
    output_dimensions = (width_f, height_f)

    #transformed_type_map = cv2.warpAffine(mask_rotated, matrix, output_dimensions)
    transformed_type_map = cv2.warpAffine(mask_rotated, matrix, output_dimensions, flags=cv2.INTER_NEAREST)

    return visium_rgb, type_map_rgb, transformed_type_map

visium_rgb, type_map_original, type_map_registered = apply_registration(path_visium, path_matrix, path_type_map)

#Plotting
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(visium_rgb)
plt.title("HE Visium")

plt.subplot(1,3,2)
plt.imshow(type_map_original)
plt.title("Original type map")

plt.subplot(1,3,3)
plt.imshow(visium_rgb)
plt.imshow(type_map_registered, alpha=0.5)
plt.title("Type map registrert")

plt.tight_layout()

overlay_path = os.path.join(dir_save, "Func116_tpmap_registered_overlay.png")
plt.savefig(overlay_path)
print("Saved overlay:", overlay_path)

plt.show()

#Result from co-registering 
registered_path = os.path.join(dir_save, "Func116_tpmap_registered.png")
cv2.imwrite(registered_path, cv2.cvtColor(type_map_registered, cv2.COLOR_RGB2BGR))
print("Saved registered type map:", registered_path)
