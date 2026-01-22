import numpy as np
import cv2

"""
This script:
Perform fine adjustments and aligments.
"""

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
 
 