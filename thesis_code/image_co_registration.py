import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

"""
This script: 
Loads the fixed Visium image and moving image from HoVer-Net results. 
Calculates the scale factor and scales the moving image to the fixed.
Manual rotation and translation. Tested different values to find the best manual alignment.
Imported function from func_co_reg.py to do fine adjustments and alignments.
Plots and saves the results.
"""

#Folders
dir_visium = '/Volumes/Expansion/Co-registration'
dir_fixed = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best'
name_list = ['Func116']


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



def fine_co_reg(fixed_rgb, moving_rgb):
    file_reduction = 0.2 #Same as in matlab to reduce size

    #Convert both to grayscale
    fixed_image = cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2GRAY)
    moving_image = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY)

    fixed_reduced = cv2.resize(fixed_image, None, fx = file_reduction, fy = file_reduction, interpolation=cv2.INTER_CUBIC)
    moving_reduced = cv2.resize(moving_image, None, fx = file_reduction, fy = file_reduction, interpolation=cv2.INTER_CUBIC)

    fixed = fixed_reduced.astype(np.uint8)
    moving = moving_reduced.astype(np.uint8)

    #ORB detector
    orb = cv2.ORB_create(5000) #5000 features
    
    #Find keypoints (kp) and descriptors (des)
    kp1, des1 = orb.detectAndCompute(moving, None) #moving
    kp2, des2 = orb.detectAndCompute(fixed, None) #fixed

    #Match features between the two images
    #Brute force matcher with Hamming distance as measurement 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #Match the sets of desciptors
    matches = matcher.match(des1, des2)

    #Sort the matches from the Hamming distance
    matches = sorted(matches, key=lambda x: x.distance)
    sorted_matches = matches[:500]
    n_matches = len(sorted_matches)
   
    moving_matrix = np.zeros((n_matches, 2))
    fixed_matrix = np.zeros((n_matches, 2))
    
    for i, m in enumerate(sorted_matches):
        moving_matrix[i] = kp1[m.queryIdx].pt
        fixed_matrix[i] = kp2[m.trainIdx].pt

    matrix, inliers = cv2.estimateAffine2D(moving_matrix, fixed_matrix, cv2.RANSAC) #Best to use estimateAffine2D! More degrees of freedom.
    #matrix, inliers = cv2.estimateAffinePartial2D(moving_matrix, fixed_matrix, cv2.RANSAC)

    #As done in matlab to resize
    matrix_resized = matrix.copy() 
    matrix_resized[0, 2] /=file_reduction
    matrix_resized[1, 2] /=file_reduction
 
    #Using matrix to transform moving image to fixed image
    height, width = fixed_rgb.shape[:2]
    registered_image = cv2.warpAffine(moving_rgb, matrix_resized, (width, height))

    #Affin transform matrix
    affine_transform_matrix = np.eye(3, dtype=np.float32)
    affine_transform_matrix[:2, :] = matrix_resized

    return registered_image, affine_transform_matrix
 

for name in (name_list):
    fixed_path = f'{dir_visium}/{name}HEVisium.tif'
    moving_path = f'{dir_fixed}/{name}_ST_HE_40x_BF_01/wsi/whole_image_complete_scaled.png'

    print(f"Fixed image from:  {fixed_path}")
    print(f"Moving image from: {moving_path}")

    #Load images
    fixed_image = cv2.imread(str(fixed_path))
    moving_image = cv2.imread(str(moving_path))

    #OpenCV uses BGR
    #Need RGB for matplotlib
    fixed_rgb = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2RGB) #Converts BGR to RGB for processing
    moving_rgb = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)

    #Scale
    scaleW, scaleH = compute_scale_factor(fixed_rgb, moving_rgb)
    moving_resized = cv2.resize(moving_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_CUBIC)

    overlay_scaled = cv2.addWeighted(fixed_rgb, 0.5, moving_resized, 0.5, 0)


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(fixed_rgb)
    plt.title(f"{name} Fixed")

    plt.subplot(1, 3, 2)
    plt.imshow(moving_resized)
    plt.title(f"{name} Moving")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay_scaled)
    plt.title(f"{name} Overlay")

    plt.tight_layout()
    plt.show()


    #Manual rotation and translation
    #Manual parameters [theta, dx, dy]
    #Func116 Manual parameters = [8, -100, 0]
    manual_angle = 8 #Rotate 8 degrees
    manual_dx = -100 #Move -100 pixels in x
    manual_dy = 0 #Move 0 pixels in y
    rotated_translated_image = manual_rotation_translation(moving_resized, manual_angle, manual_dx, manual_dy) #Rotation and translation
    overlay_manual = cv2.addWeighted(fixed_rgb, 0.5, rotated_translated_image, 0.5, 0)
    plt.imshow(overlay_manual)
    plt.title("Manual co-registrering")
    plt.show()


    #Func co reg
    #Fine adjustments
    moving_registered, affine_matrix = fine_co_reg(fixed_rgb, rotated_translated_image)
    overlay_fine = cv2.addWeighted(fixed_rgb, 0.5, moving_registered, 0.5, 0)
    plt.imshow(overlay_fine)
    plt.title("Finjustert co-registrering")
    plt.show()


    save_dir = "/Volumes/Expansion/biopsy_results/pannuke/40x/co_registration/"
    os.makedirs(save_dir, exist_ok=True)

    #Save overlays
    plt.imsave(os.path.join(save_dir, f"{name}_overlay_grov.png"), overlay_scaled)
    plt.imsave(os.path.join(save_dir, f"{name}_overlay_manual.png"), overlay_manual)
    plt.imsave(os.path.join(save_dir, f"{name}_overlay_fine.png"), overlay_fine)

    #Save affine transform matrix
    np.save(os.path.join(save_dir, f"{name}_affine_transform.npy"), affine_matrix)

    #Save manual rotated and translated image
    rotated_translated_path = os.path.join(save_dir, f"{name}_manual_rotation_translation.png")
    cv2.imwrite(rotated_translated_path, cv2.cvtColor(rotated_translated_image, cv2.COLOR_RGB2BGR))

    #Save moving registered image
    registered_path = os.path.join(save_dir, f"{name}_moving_registered.png")
    cv2.imwrite(registered_path, cv2.cvtColor(moving_registered, cv2.COLOR_RGB2BGR))

