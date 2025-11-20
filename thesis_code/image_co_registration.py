#Image Co-registration

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#import sys 
#script_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(script_dir)
#from func_co_reg import func_co_reg

"""
The image_co_registration.py script: 
Loads fixed and moving image. 
Calculates the scaling factor.
Scales the moving image to the size as fixed.
Manual rotation and translation.
Plots the results.
"""

#Folders
dir_visium = '/Volumes/Expansion/Co-registration'
name_list = ['Func116']
dir_dig_path = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best'

def tissue_bbox(rgb_image, thresh=240):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    mask = gray < thresh
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    return x_min, y_min, x_max, y_max

#Computing scaling factor
def compute_scaling_factor(fixed_image, moving_image):  # fixed: Visium, moving: HoverNet results
    height_f, width_f = fixed_image.shape[:2]
    height_m, width_m = moving_image.shape[:2]

    scaleH = height_f / height_m
    scaleW = width_f / width_m

    #scale_factor = np.mean([scaleH, scaleW])
    #return scale_factor
    return scaleW, scaleH


#Rotation and translation
def func_manual_rotation(image, angle, tx, ty):
    #Rotating by an arbitrary angle
    #Get image dimensions
    height, width = image.shape[:2] #take the two first values from (H, W, C

    #Define the rotation parameters
    center = (width // 2, height // 2) #Center of rotation

    rotation_angle = angle #Chosen rotation angle in degrees (counter-clockwise)
    scale = 1.0 #Scaling factor

    #Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)

    # Perform the affine transformation (rotation)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    #Translation values
    translate_x = tx
    translate_y = ty
    translation_matrix = np.float32([[1, 0, translate_x],[0, 1, translate_y]])

    translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))

    return translated_image


"""
The func_co_reg script:
Takes in the images from image co registration.
Then does fine adjustments and aligments.
"""

def func_co_reg(fixed_rgb, moving_rgb):
    file_reduction = 0.2 #reduce size
    height, width = fixed_rgb.shape[:2]

    #Convert both to grayscale
    fixed_image = cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2GRAY)
    moving_image = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY)

    fixed_reduced = cv2.resize(fixed_image, None, fx = file_reduction, fy = file_reduction, interpolation=cv2.INTER_CUBIC)
    moving_reduced = cv2.resize(moving_image, None, fx = file_reduction, fy = file_reduction, interpolation=cv2.INTER_CUBIC)

    fixed = fixed_reduced.astype(np.uint8)
    moving = moving_reduced.astype(np.uint8)

    #ORB detector with 5000 features
    orb = cv2.ORB_create(5000)
    
    #Find keypoints (kp) and descriptors (des)
    #The first arg is the image, second arg is the mask
    kp1, des1 = orb.detectAndCompute(moving, None) #moving
    kp2, des2 = orb.detectAndCompute(fixed, None) #fixed

    #Match features between the two images
    #Brute force matcher with Hamming distance as measurement mode
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


    #matrix, mask = cv2.estimateAffine2D(moving_matrix, fixed_matrix, cv2.RANSAC)
    matrix, inliers = cv2.estimateAffinePartial2D(moving_matrix, fixed_matrix, method=cv2.RANSAC)

#     matrix, inliers = cv2.estimateAffinePartial2D(
#     moving_matrix,
#     fixed_matrix,
#     method=cv2.RANSAC,
#     ransacReprojThreshold=3.0,
#     maxIters=5000,
#     confidence=0.99,
#     refineIters=10,
# )

    matrix_resized = matrix.copy()
    matrix_resized[0, 2] /=file_reduction
    matrix_resized[1, 2] /=file_reduction

    #Using matrix to transform moving image to fixed image
    registered_image = cv2.warpAffine(moving_rgb, matrix_resized, (width, height))

    rot_matrix = np.eye(3, dtype=np.float32)
    rot_matrix[:2, :] = matrix_resized


    return registered_image, matrix_resized
 

for name in (name_list):
    fixed_path = f'{dir_visium}/{name}HEVisium.tif'
    moving_path = f'{dir_dig_path}/{name}_ST_HE_40x_BF_01/wsi/whole_image_complete_scaled.png'

    print(f"Fixed image from:  {fixed_path}")
    print(f"Moving image from: {moving_path}")

    fixed_image = cv2.imread(str(fixed_path))
    moving_image = cv2.imread(str(moving_path))

    #OpenCV use BGR
    #Need RGB for matplotlib
    fixed_rgb = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2RGB) #Converts BGR to RGB for processing
    moving_rgb = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)

    #scale = compute_scaling_factor(fixed_rgb, moving_rgb)
    #moving_scaled = cv2.resize(moving_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    #scaleW, scaleH = compute_scaling_factor(fixed_rgb, moving_rgb)
    #moving_scaled = cv2.resize(moving_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_CUBIC)

    #height_f, width_f = fixed_rgb.shape[:2]
    #moving_resized = cv2.resize(moving_scaled, (width_f, height_f), interpolation=cv2.INTER_CUBIC)

    fx1, fy1, fx2, fy2 = tissue_bbox(fixed_rgb)
    mx1, my1, mx2, my2 = tissue_bbox(moving_rgb)

    fixed_crop  = fixed_rgb[fy1:fy2, fx1:fx2]
    moving_crop = moving_rgb[my1:my2, mx1:mx2]

    scaleH = fixed_crop.shape[0] / moving_crop.shape[0]
    scaleW = fixed_crop.shape[1] / moving_crop.shape[1]

    moving_scaled = cv2.resize(moving_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_CUBIC)

    height_f, width_f = fixed_rgb.shape[:2]
    moving_resized = cv2.resize(moving_scaled, (width_f, height_f), interpolation=cv2.INTER_CUBIC)

    overlay = cv2.addWeighted(fixed_rgb, 0.5, moving_resized, 0.5, 0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(fixed_rgb)
    plt.title(f"{name} Fixed")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(moving_resized)
    plt.title(f"{name} Moving")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"{name} Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    #Manual rotation
    #Manual rotation parameters [theta, dx, dy]
    manual_rotation = [8, -100, 0]  #Func116

    rotating_image = func_manual_rotation(moving_resized, *manual_rotation)
 
    overlay2 = cv2.addWeighted(fixed_rgb, 0.5, rotating_image, 0.5, 0)
    plt.imshow(overlay2)
    plt.show()

    #Func co reg
    reg_image, rot_matrix = func_co_reg(fixed_rgb, rotating_image)

    overlay_fine = cv2.addWeighted(fixed_rgb, 0.5, reg_image, 0.5, 0)
    plt.imshow(overlay_fine)
    plt.title("Finjustert co-registrering")
    plt.axis("off")
    plt.show()

 

    save_dir = "/Volumes/Expansion/biopsy_results/pannuke/40x/co_reg_opencv_testing_box_extra/"
    os.makedirs(save_dir, exist_ok=True)

    #save grove overlay
    plt.imsave(os.path.join(save_dir, f"{name}_overlay_grov.png"), overlay)
    plt.imsave(os.path.join(save_dir, f"{name}_overlay_manual.png"), overlay2)

    #save finjustert overlay
    plt.imsave(os.path.join(save_dir, f"{name}_overlay_fine.png"), overlay_fine)

    #save dregistered image (moving to aligned to fixed)
    plt.imsave(os.path.join(save_dir, f"{name}_moving_registered.png"), reg_image)

    #save matrix
    np.save(os.path.join(save_dir, f"{name}_rot_matrix.npy"), rot_matrix)

    #save rotating image
    rot_png_path = os.path.join(save_dir, f"{name}_rotating_manual.png")
    cv2.imwrite(rot_png_path, cv2.cvtColor(rotating_image, cv2.COLOR_RGB2BGR))
