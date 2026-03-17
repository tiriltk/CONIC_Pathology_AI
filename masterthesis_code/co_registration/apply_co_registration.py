import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Apply co-registration following Matlab and functions from image_co_registration.py.
Have co-registered the biopsy and now wants to use the same transformation on the type map.
"""

#File paths
path_visium = "/Volumes/Expansion/Co-registration/Func116HEVisium.tif" #Visium
path_type_map = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func116_ST_HE_40x_BF_01/wsi_tp_results/Func116_tpmap_scaled.png" #Type map
#path_type_map = "/Volumes/Expansion/biopsy_results/conic/20x/output_fill/Func116_ST_HE_20x_BF_01/wsi_border_type_map/bordered_type_map.png"
path_matrix = "/Volumes/Expansion/biopsy_results/pannuke/40x/co_registration/co_reg_biopsy/Func116_affine_transform.npy" #Affine matrix
dir_save = "Volumes/Expansion/biopsy_results/pannuke/40x/co_registration/" #Saving directory
os.makedirs(dir_save, exist_ok=True)

#Functions from image_co_registation.py:
#Compute scale factor
#Compute scale factor
def compute_scale_factor(fixed_image, moving_image): 
    height_fixed, width_fixed = fixed_image.shape[:2]
    height_moving, width_moving = moving_image.shape[:2]
    scaleH = height_fixed / height_moving
    scaleW = width_fixed / width_moving
    return scaleW, scaleH

#Manual rotation and translation on moving HoVer-Net image
def manual_rotation_translation(image, angle, tx, ty):
    height, width = image.shape[:2] #Get image dimensions
    
    center = (width // 2, height // 2) #Center coordinates 
    rotation_angle = angle #Selected rotation angle in degrees where positive angle is counter clockwise
    scale = 1.0 #Scaling factor no scaling
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale) #Rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height)) #Apply rotation with warpAffine function
    
    translation_matrix = np.float32([[1, 0, tx],[0, 1, ty]]) #Translation matrix
    rotated_translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height)) #Apply translation 

    return rotated_translated_image

def apply_registration(visium_image_path, affine_matrix_path, type_map_path):
    visium = cv2.imread(visium_image_path) #Load fixed_image bgr
    visium_rgb = cv2.cvtColor(visium, cv2.COLOR_BGR2RGB)
    type_map = cv2.imread(type_map_path) ##Load type_map bgr
    type_map_rgb = cv2.cvtColor(type_map, cv2.COLOR_BGR2RGB)

    scaleW, scaleH = compute_scale_factor(visium_rgb, type_map_rgb) #Scale
    type_map_resized = cv2.resize(type_map_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_NEAREST) #best for binary maps

    #Manual rotation and translation Func116 manual parameters = [8, -100, 0]
    manual_angle = 8 #Rotate 8 degrees
    manual_dx = -100 #Move -100 pixels in x
    manual_dy = 0 #Move 0 pixels in y
    mask_rotated = manual_rotation_translation(type_map_resized, manual_angle, manual_dx, manual_dy)

    #Load affine matrix
    matrix_full = np.load(affine_matrix_path) #3x3
    matrix = matrix_full[:2, :] #2x3 required by warpAffine
    height_f, width_f = visium_rgb.shape[:2] 
    output_dimensions = (width_f, height_f) #Dimensions for the output image

    transformed_type_map = cv2.warpAffine(mask_rotated, matrix, output_dimensions, flags=cv2.INTER_NEAREST) #Apply transformation
    #transformed_type_map = cv2.warpAffine(mask_rotated, matrix, output_dimensions)

    return visium_rgb, type_map_rgb, transformed_type_map

visium_rgb, type_map_original, type_map_registered = apply_registration(path_visium, path_matrix, path_type_map)

#Plot
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

overlay_path = os.path.join(dir_save, "Func116_tpmap_registered_overlay.png") #Save
plt.savefig(overlay_path)
print("Saved:", overlay_path)
plt.show()

registered_path = os.path.join(dir_save, "Func116_tpmap_registered.png") #Save result from co-registering 
cv2.imwrite(registered_path, cv2.cvtColor(type_map_registered, cv2.COLOR_RGB2BGR))
print("Saved registered type map:", registered_path)
