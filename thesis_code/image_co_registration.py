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
From func_co_reg fine adjustments and alignments.
Plots and saves the results.
"""

#Folders
dir_visium = '/Volumes/Expansion/Co-registration'
name_list = ['Func116']
dir_dig_path = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best'


#Find the color of the background pixels to select treshhold value for separating the biopsy from background
def pixel_color(image_path):
    img_gray = cv2.imread(image_path, 0) #Image in grayscale

    #Coordinates to pick pixel from
    x = 20
    y = 20
    pixel_value = img_gray[y, x]

    print(f"The pixel value: {pixel_value}")

    #The pixel value for fixed: (10x10) = 238, (5x5) = 237, (20x20) = 239
    #The pixel value for moving: (10x20) = 240,(5x5) = 240, (20x20) = 241

    return pixel_value


#Select a tight box around the circular biopsy to scale better as the biopsies have different sizes
#This gives better co-registartion than using the whole image with lots of background
def biopsy_mask(rgb_image, thresh=230):
    gray_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    mask = gray_img < thresh
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()  #top and bottom
    x_min, x_max = xs.min(), xs.max()  #left and right
    return x_min, y_min, x_max, y_max


#Compute scaling factor
def compute_scaling_factor(fixed_image, moving_image):  #Fixed: Visium, moving: HoverNet results
    height_fixed, width_fixed = fixed_image.shape[:2]
    height_moving, width_moving = moving_image.shape[:2]

    scaleH = height_fixed / height_moving
    scaleW = width_fixed / width_moving

    #scale_factor = np.mean([scaleH, scaleW])
    #return scale_factor

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


    matrix, inliers = cv2.estimateAffine2D(moving_matrix, fixed_matrix, cv2.RANSAC) #Best to use estimateAffine2D!
    #matrix, inliers = cv2.estimateAffinePartial2D(moving_matrix, fixed_matrix, cv2.RANSAC)

    matrix_resized = matrix.copy()
    matrix_resized[0, 2] /=file_reduction
    matrix_resized[1, 2] /=file_reduction
 

    #Using matrix to transform moving image to fixed image
    registered_image = cv2.warpAffine(moving_rgb, matrix_resized, (width, height))

    #Affin transform-matrise
    affine_transform_matrix = np.eye(3, dtype=np.float32)
    affine_transform_matrix[:2, :] = matrix_resized

    return registered_image, affine_transform_matrix
 

for name in (name_list):
    fixed_path = f'{dir_visium}/{name}HEVisium.tif'
    moving_path = f'{dir_dig_path}/{name}_ST_HE_40x_BF_01/wsi/whole_image_complete_scaled.png'

    print(f"Fixed image from:  {fixed_path}")
    print(f"Moving image from: {moving_path}")

    fixed_image = cv2.imread(str(fixed_path))
    moving_image = cv2.imread(str(moving_path))

    #pixel_color(fixed_path)
    #pixel_color(moving_path)    

    #OpenCV uses BGR
    #Need RGB for matplotlib
    fixed_rgb = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2RGB) #Converts BGR to RGB for processing
    moving_rgb = cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB)

    #scale = compute_scaling_factor(fixed_rgb, moving_rgb)
    #moving_scaled = cv2.resize(moving_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    #scaleW, scaleH = compute_scaling_factor(fixed_rgb, moving_rgb)
    #moving_scaled = cv2.resize(moving_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_CUBIC)

    #height_f, width_f = fixed_rgb.shape[:2]
    #moving_resized = cv2.resize(moving_scaled, (width_f, height_f), interpolation=cv2.INTER_CUBIC)

    #Biopsy mask finds box around the biopsies
    fx1, fy1, fx2, fy2 = biopsy_mask(fixed_rgb) #fixed box
    mx1, my1, mx2, my2 = biopsy_mask(moving_rgb) #moving box

    #Crops out the box with the biopsies to make registration better
    fixed_crop  = fixed_rgb[fy1:fy2, fx1:fx2] 
    moving_crop = moving_rgb[my1:my2, mx1:mx2]

    #Scaling the cropped biopsy
    height_f = fixed_crop.shape[0]
    width_f  = fixed_crop.shape[1]

    height_m = moving_crop.shape[0]
    width_m  = moving_crop.shape[1]

    scaleH = height_f / height_m
    scaleW = width_f  / width_m

    #Scale whole moving image with the scaling factors
    moving_scaled = cv2.resize(moving_rgb, None, fx=scaleW, fy=scaleH, interpolation=cv2.INTER_CUBIC)

    #Moving to the same dimensions as fixed
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
    manual_rotation = [8, -100, 0]  #Func116, rotate 8 degrees, move -100 pixels in x, 0 pixels in y

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


    save_dir = "/Volumes/Expansion/biopsy_results/pannuke/40x/co_reg_best/"
    os.makedirs(save_dir, exist_ok=True)

    #save grove overlay
    plt.imsave(os.path.join(save_dir, f"{name}_overlay_grov.png"), overlay)
    plt.imsave(os.path.join(save_dir, f"{name}_overlay_manual.png"), overlay2)

    #save finjustert overlay
    plt.imsave(os.path.join(save_dir, f"{name}_overlay_fine.png"), overlay_fine)

    #save dregistered image (moving to aligned to fixed)
    plt.imsave(os.path.join(save_dir, f"{name}_moving_registered.png"), reg_image)

    #save matrix
    np.save(os.path.join(save_dir, f"{name}_affine_transform.npy"), rot_matrix)

    #save rotating image
    rot_png_path = os.path.join(save_dir, f"{name}_rotating_manual.png")
    cv2.imwrite(rot_png_path, cv2.cvtColor(rotating_image, cv2.COLOR_RGB2BGR))
