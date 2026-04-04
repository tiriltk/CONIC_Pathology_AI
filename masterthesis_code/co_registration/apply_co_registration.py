import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Apply co-registration following Matlab and functions from image_co_registration.py.
Have co-registered the biopsy and now wants to use the same transformation on the type map.
"""

path_visium = "/Volumes/Expansion/Co-registration/Func116HEVisium.tif" #Visium
path_type_map = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func116_ST_HE_40x_BF_01/wsi_tp_results/Func116_tpmap_scaled.png" #Type map
path_matrix = "/Volumes/Expansion/biopsy_results/pannuke/40x/co_registration/Func116_affine_transform.npy" #Affine matrix
dir_save = "/Volumes/Expansion/biopsy_results/pannuke/40x/co_registration/" #Saving directory
os.makedirs(dir_save, exist_ok=True)

#Functions copied from image_co_registation.py:
#Compute scaling factors
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
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0) #Rotation matrix, selected rotation angle in degrees where positive angle is counter clockwise
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height)) #Apply rotation with warpAffine function

    translation_matrix = np.float32([[1, 0, tx],[0, 1, ty]]) #Translation matrix
    manual_aligned_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height)) #Apply translation 
    return manual_aligned_image

def apply_registration(visium_image_path, affine_matrix_path, type_map_path):
    visium = cv2.imread(visium_image_path) #Load fixed_image bgr
    visium_rgb = cv2.cvtColor(visium, cv2.COLOR_BGR2RGB)
    type_map = cv2.imread(type_map_path) ##Load type_map bgr
    type_map_rgb = cv2.cvtColor(type_map, cv2.COLOR_BGR2RGB)

    scaleW, scaleH = compute_scale_factor(visium_rgb, type_map_rgb) #Scale
    type_map_resized = cv2.resize(type_map_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_NEAREST) #using nearest, best for type maps

    #Manual rotation and translation Func116 manual parameters: Rotate 8 degrees, move -100 pixels in x, move 0 pixels in y
    type_map_manual_aligned = manual_rotation_translation(type_map_resized, 8, -100, 0)

    matrix_full = np.load(affine_matrix_path) #Load stored affine matrix 3x3
    matrix = matrix_full[:2, :] #2x3 required by warpAffine
    height_f, width_f = visium_rgb.shape[:2] 
    output_dimensions = (width_f, height_f) #Dimensions for the output image
    transformed_type_map = cv2.warpAffine(type_map_manual_aligned, matrix, output_dimensions, flags=cv2.INTER_NEAREST) #Apply transformation

    return visium_rgb, type_map_rgb, transformed_type_map

visium_rgb, type_map_original, type_map_registered = apply_registration(path_visium, path_matrix, path_type_map)

plt.imshow(visium_rgb)
plt.imshow(type_map_registered, alpha=0.5)
plt.title("Type map registrert")

overlay_path = os.path.join(dir_save, "Func116_typemap_registered_overlay.png") #Save overlay
plt.savefig(overlay_path)
print("Saved:", overlay_path)
plt.show()

registered_path = os.path.join(dir_save, "Func116_typemap_registered.png") #Save co registered result
cv2.imwrite(registered_path, cv2.cvtColor(type_map_registered, cv2.COLOR_RGB2BGR))
print("Saved:", registered_path)
