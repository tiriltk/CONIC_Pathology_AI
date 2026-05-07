import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

"""
Co-registration following Matlab code and OpenCv examples. Fixed image is Visium and moving image is HoVerNet results. 
Calculates the scale factor and scales the moving image to the fixed. 
Apply manual rotation and translation by testing different values.
Fine adjustments and alignments using OpenCV functions. Plots and saves the results.
"""

#File paths
fixed_path = "/Volumes/Expansion/Co-registration/Func116HEVisium.tif" #Fixed image path
moving_path = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func116_ST_HE_40x_BF_01/wsi/whole_image_complete_scaled.png" #Moving image path
save_dir = "/Volumes/Expansion/biopsy_results/pannuke/40x/co_registration/" #Save path
os.makedirs(save_dir, exist_ok=True)

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
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0) #Rotation matrix with selected rotation angle in degrees where positive angle is counter clockwise
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height)) #Apply rotation with warpAffine function

    translation_matrix = np.float32([[1, 0, tx],[0, 1, ty]]) #Translation matrix
    manual_aligned_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height)) #Apply translation 
    return manual_aligned_image

#Fine and automatic co registration 
def fine_co_reg(fixed_rgb, moving_manual_rgb):
    fixed_image = cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2GRAY) #Convert to grayscale
    moving_image = cv2.cvtColor(moving_manual_rgb, cv2.COLOR_RGB2GRAY)

    file_reduction = 0.2 #Same as used in Matlab to reduce image size
    fixed_reduced = cv2.resize(fixed_image, None, fx = file_reduction, fy = file_reduction, interpolation=cv2.INTER_CUBIC) #Cubic interpolation for high quality
    moving_reduced = cv2.resize(moving_image, None, fx = file_reduction, fy = file_reduction, interpolation=cv2.INTER_CUBIC)
    fixed = fixed_reduced.astype(np.uint8)
    moving = moving_reduced.astype(np.uint8)

    orb = cv2.ORB_create(5000) #ORB feature detector with number of features
    kp1, des1 = orb.detectAndCompute(moving, None) #Find keypoints (kp) and descriptors (des)
    kp2, des2 = orb.detectAndCompute(fixed, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #Matcher 
    matches = matcher.match(des1, des2) #Match the descriptors
    matches = sorted(matches, key=lambda x: x.distance) #Sort matches from distances
    sorted_matches = matches[:500] #Selected the best matches

    n_matches = len(sorted_matches)
    moving_points = np.zeros((n_matches, 2)) #Define matrices to save coordinates
    fixed_points = np.zeros((n_matches, 2))
    for i, m in enumerate(sorted_matches):
        moving_points[i] = kp1[m.queryIdx].pt
        fixed_points[i] = kp2[m.trainIdx].pt

    matrix, inliers = cv2.estimateAffine2D(moving_points, fixed_points, cv2.RANSAC) #Best results with estimateAffine2D
    matrix_resized = matrix.copy() 
    matrix_resized[0, 2] /=file_reduction #As used in Matlab to resize back translation
    matrix_resized[1, 2] /=file_reduction
    height, width = fixed_rgb.shape[:2]
    registered_image = cv2.warpAffine(moving_manual_rgb, matrix_resized, (width, height)) #Apply affine transformation to transform moving image to fixed image

    affine_transform_matrix = np.eye(3, dtype=np.float32) #As done in Matlab to save affine transformation matrix 3x3
    affine_transform_matrix[:2, :] = matrix_resized
    return registered_image, affine_transform_matrix

#Load images
fixed_image = cv2.imread(str(fixed_path)) 
moving_image = cv2.imread(str(moving_path))
fixed_rgb = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2RGB) #Convert image from BGR to RGB
moving_rgb = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)

scaleW, scaleH = compute_scale_factor(fixed_rgb, moving_rgb) #Scale moving image to fixed image
moving_resized = cv2.resize(moving_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_CUBIC) #Scale with interpolation best for images high quality
overlay_scaled = cv2.addWeighted(fixed_rgb, 0.5, moving_resized, 0.5, 0) #Overlay

#Func116 sample manual parameters: Rotate 8 degrees, move -100 pixels in x, move 0 pixels in y
manual_aligned_image = manual_rotation_translation(moving_resized, 8, -100, 0) #Apply manual rotation translation
overlay_manual = cv2.addWeighted(fixed_rgb, 0.5, manual_aligned_image, 0.5, 0) #Overlay
plt.imshow(overlay_manual)
plt.title("Manual registration")
plt.show()

#Fine adjustments
moving_fine_registered, affine_matrix = fine_co_reg(fixed_rgb, manual_aligned_image)
overlay_fine = cv2.addWeighted(fixed_rgb, 0.5, moving_fine_registered, 0.5, 0)
plt.imshow(overlay_fine)
plt.title("Fine adjusted registration")
plt.show()

#Save results
plt.imsave(os.path.join(save_dir, "Func116_overlay_scaled.png"), overlay_scaled) #Save overlays
plt.imsave(os.path.join(save_dir, "Func116_overlay_manual.png"), overlay_manual)
plt.imsave(os.path.join(save_dir, "Func116_overlay_fine.png"), overlay_fine)

np.save(os.path.join(save_dir, "Func116_affine_transform.npy"), affine_matrix) #Save affine transform matrix
manual_aligned_path = os.path.join(save_dir, "Func116_manual_aligned.png") #Save manual aligned image
cv2.imwrite(manual_aligned_path, cv2.cvtColor(manual_aligned_image, cv2.COLOR_RGB2BGR))
registered_path = os.path.join(save_dir, "Func116_co_registered.png") #Save final co registered image
cv2.imwrite(registered_path, cv2.cvtColor(moving_fine_registered, cv2.COLOR_RGB2BGR))