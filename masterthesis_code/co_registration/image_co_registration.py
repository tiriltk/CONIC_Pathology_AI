import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

"""
Co-registration following Matlab code and OpenCv examples. Fixed image is Visium and moving image is HoVerNet results. 
Calculates the scale factor and scales the moving image to the fixed. Manual rotation and translation.
Fine adjustments and alignments using OpenCV functions. Plots and saves the results.
"""

#File paths
fixed_path = "/Volumes/Expansion/Co-registration/Func116HEVisium.tif" #Fixed image path Visium Moving
moving_path = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func116_ST_HE_40x_BF_01/wsi/whole_image_complete_scaled.png" #Moving image path HoVer-Net results
save_dir = "/Volumes/Expansion/biopsy_results/pannuke/40x/co_registration/" #Save path
os.makedirs(save_dir, exist_ok=True)

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

def fine_co_reg(fixed_rgb, moving_rgb):
    fixed_image = cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2GRAY) #Convert to grayscale
    moving_image = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY)

    file_reduction = 0.2 #Same as in Matlab to reduce size
    fixed_reduced = cv2.resize(fixed_image, None, fx = file_reduction, fy = file_reduction, interpolation=cv2.INTER_CUBIC) #best for images high quality
    moving_reduced = cv2.resize(moving_image, None, fx = file_reduction, fy = file_reduction, interpolation=cv2.INTER_CUBIC)
    fixed = fixed_reduced.astype(np.uint8)
    moving = moving_reduced.astype(np.uint8)

    #ORB detector, tried different values for feaures
    orb = cv2.ORB_create(5000) 
    kp1, des1 = orb.detectAndCompute(moving, None) #Find keypoints (kp) and descriptors (des)
    kp2, des2 = orb.detectAndCompute(fixed, None)

    #Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #Brute force matcher
    matches = matcher.match(des1, des2) #Match the sets of desciptors
    matches = sorted(matches, key=lambda x: x.distance) #Sort the matches from distances
    sorted_matches = matches[:500] #Tried different values

    n_matches = len(sorted_matches)
    moving_matrix = np.zeros((n_matches, 2)) #Define matrixes to save coordinates
    fixed_matrix = np.zeros((n_matches, 2))
    for i, m in enumerate(sorted_matches):
        moving_matrix[i] = kp1[m.queryIdx].pt
        fixed_matrix[i] = kp2[m.trainIdx].pt

    matrix, inliers = cv2.estimateAffine2D(moving_matrix, fixed_matrix, cv2.RANSAC) #Tried different, best to use estimateAffine2D! More degrees of freedom
    matrix_resized = matrix.copy() #As done in matlab to resize
    matrix_resized[0, 2] /=file_reduction
    matrix_resized[1, 2] /=file_reduction
 
    height, width = fixed_rgb.shape[:2]
    registered_image = cv2.warpAffine(moving_rgb, matrix_resized, (width, height)) #Using matrix to transform moving image to fixed image
    affine_transform_matrix = np.eye(3, dtype=np.float32) #Affin transform matrix 3x3
    affine_transform_matrix[:2, :] = matrix_resized

    return registered_image, affine_transform_matrix

#Load images
fixed_image = cv2.imread(str(fixed_path))
moving_image = cv2.imread(str(moving_path))
fixed_rgb = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2RGB) #Converts BGR to RGB
moving_rgb = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)

scaleW, scaleH = compute_scale_factor(fixed_rgb, moving_rgb) #Scale
moving_resized = cv2.resize(moving_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_CUBIC) #best for images high quality
overlay_scaled = cv2.addWeighted(fixed_rgb, 0.5, moving_resized, 0.5, 0)

#Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(fixed_rgb)
plt.title("Fixed")

plt.subplot(1, 3, 2)
plt.imshow(moving_resized)
plt.title("Moving")

plt.subplot(1, 3, 3)
plt.imshow(overlay_scaled)
plt.title("Overlay")
plt.tight_layout()
plt.show()

#Manual rotation and translation Func116 manual parameters = [8, -100, 0]
manual_angle = 8 #Rotate 8 degrees
manual_dx = -100 #Move -100 pixels in x
manual_dy = 0 #Move 0 pixels in y
rotated_translated_image = manual_rotation_translation(moving_resized, manual_angle, manual_dx, manual_dy)
overlay_manual = cv2.addWeighted(fixed_rgb, 0.5, rotated_translated_image, 0.5, 0)
plt.imshow(overlay_manual)
plt.title("Manual registration")
plt.show()

#Fine adjustments
moving_registered, affine_matrix = fine_co_reg(fixed_rgb, rotated_translated_image)
overlay_fine = cv2.addWeighted(fixed_rgb, 0.5, moving_registered, 0.5, 0)
plt.imshow(overlay_fine)
plt.title("Fine adjusted registration")
plt.show()

plt.imsave(os.path.join(save_dir, "Func116_overlay_grov.png"), overlay_scaled) #Save overlays
plt.imsave(os.path.join(save_dir, "Func116_overlay_manual.png"), overlay_manual)
plt.imsave(os.path.join(save_dir, "Func116_overlay_fine.png"), overlay_fine)

np.save(os.path.join(save_dir, "Func116_affine_transform.npy"), affine_matrix) #Save affine transform matrix
rotated_translated_path = os.path.join(save_dir, "Func116_manual_rotation_translation.png") #Save manual rotated and translated image
cv2.imwrite(rotated_translated_path, cv2.cvtColor(rotated_translated_image, cv2.COLOR_RGB2BGR))
registered_path = os.path.join(save_dir, "Func116_moving_registered.png") #Save moving registered image
cv2.imwrite(registered_path, cv2.cvtColor(moving_registered, cv2.COLOR_RGB2BGR))