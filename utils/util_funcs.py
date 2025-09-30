import cv2
import numpy as np
import os
import random
import math
import pandas as pd
from scipy.ndimage import measurements
from skimage import morphology as morph
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import shutil

def rm_n_mkdir(dir_path):
    """Remove and then make a new directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def draw_dilation(img, instance_mask, instance_type, label_colors, nuclei_marker, dataset):
    # Change type of image if necessarily
    if img.dtype == np.float64:
        img = img.astype(np.uint8)             # convert image to uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert image from rgb to bgr
    img_overlay = img.copy()

    # instance_list = np.unique(instance_mask)[1:] # drop background, but find all other instances including nuclei with no label
    # for instance in instance_list:
    #     binary_map = np.zeros_like(img, dtype=np.uint8)

    #     indexes = np.where(instance_mask == instance)

    #     binary_map[indexes] = 255
    #     kernal = np.ones((5, 5), np.uint8)
    #     dilation = cv2.dilate(binary_map, kernal, iterations=1)
    #     inst_pixels_dilated = np.where((dilation == [255, 255, 255]).all(axis=2))

    #     instance_tp = np.unique(instance_type[indexes])
    #     assert len(instance_tp) == 1, "wrong instance type correspondence! "
    #     if instance_tp[0] == 0: # If instance has no type label, exclude it from overlay
    #         continue
    #     else:
    #         img_overlay[inst_pixels_dilated] = label_colors[(instance_tp[0]-1)] # (instance_tp[0]-1) corrects for no label nuclei being removed from overlay
    #         img_overlay[indexes] = img[indexes]


    n_classes = 5 if dataset == "pannuke" else 6
    pixel_per_type = {str(i): 0 for i in range(1, n_classes + 1)}

    #conic
    #pixel_per_type = {"1": 0,
            #"2": 0,
            #"3": 0,
            #"4": 0,
            #"5": 0,
            #"6": 0}
    
    if nuclei_marker == "fill":
        instance_list = np.unique(instance_mask)[1:] # drop background, but find all other instances including nuclei with no label
        for instance in instance_list:
            binary_map = np.zeros_like(img, dtype=np.uint8)

            indexes = np.where(instance_mask == instance)
            
            binary_map[indexes] = 255
             # Count the number of pixels that are 255
            count_255_pixels = np.sum(binary_map == 255)
            
            # Output the count
            kernal = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(binary_map, kernal, iterations=1)
            inst_pixels_dilated = np.where((dilation == [255, 255, 255]).all(axis=2))
            instance_tp = np.unique(instance_type[indexes])
            # print(f'Instance {instance} of type {instance_tp[0]} has {count_255_pixels} pixels with value 255.')
            assert len(instance_tp) == 1, "wrong instance type correspondence! "
            if instance_tp[0] == 0: # If instance has no type label, exclude it from overlay
                continue
            else:
                pixel_per_type[str(instance_tp[0])] += count_255_pixels
                img_overlay[indexes] = label_colors[(instance_tp[0]-1)] # (instance_tp[0]-1) corrects for no label nuclei being removed from overlay       
    
    elif nuclei_marker == "border":
        instance_list = np.unique(instance_mask)[1:] # drop background, but find all other instances including nuclei with no label
        for instance in instance_list:
            binary_map = np.zeros_like(img, dtype=np.uint8)

            indexes = np.where(instance_mask == instance)

            binary_map[indexes] = 255
             # Count the number of pixels that are 255
            count_255_pixels = np.sum(binary_map == 255)

            kernal = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(binary_map, kernal, iterations=1)
            inst_pixels_dilated = np.where((dilation == [255, 255, 255]).all(axis=2))
            instance_tp = np.unique(instance_type[indexes])
            assert len(instance_tp) == 1, "wrong instance type correspondence! "
            if instance_tp[0] == 0: # If instance has no type label, exclude it from overlay
                continue
            else:
                pixel_per_type[str(instance_tp[0])] += count_255_pixels
                img_overlay[inst_pixels_dilated] = label_colors[(instance_tp[0]-1)] # (instance_tp[0]-1) corrects for no label nuclei being removed from overlay
                img_overlay[indexes] = img[indexes]

    #for chosen dataset
    if dataset == "pannuke":
        key_mapping = {
            "1": "neoplastic",
            "2": "inflammatory",
            "3": "connective",
            "4": "dead",
            "5": "epithelial",
        }

    elif dataset == "conic":
        key_mapping = {"1" : "neutrophil",
                "2" : "epithelial",
                "3" : "lymphocyte",
                "4" : "plasma",
                "5" : "eosinophil",
                "6" : "connective"}

    # Create a new dictionary with the new keys
    new_dict = {key_mapping.get(k, k): v for k, v in pixel_per_type.items()}

    # pixel_count_df = pd.DataFrame(new_dict, index = ["pixels"])
    return img_overlay, new_dict

def draw_dilation_monusac(img, instance_mask):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_overlay = img.copy()
    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    instance_list = np.unique(instance_mask)[1:]
    for instance in instance_list:
        binary_map = np.zeros_like(img, dtype=np.uint8)

        indexes = np.where(instance_mask == instance)

        binary_map[indexes] = 255
        kernal = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(binary_map, kernal, iterations=1)
        inst_pixels_dilated = np.where((dilation == [255, 255, 255]).all(axis=2))

        img_overlay[inst_pixels_dilated] = color
        img_overlay[indexes] = img[indexes]
    return img_overlay


def visualize(imgs, ann, pred, output_dir, dataset, nuclei_marker):
    
    # Decide colors for visualization of nuclei based on datset used
    if dataset=="conic":
        print(f"Dataset in visualize: {dataset}")
        # BGR values used in color
        colors = [[0  ,   0,   255], [0,   0,   0], [255  ,   255, 0], 
                [255 ,   0, 0], [255, 0,   255], [0, 255,   0], [0, 255, 255]]
        # color order nuclei: nolable (red), neutrphil (black), epithelial (cyan), lymphocyte (dark blue), plasma (magenta), eosinophil (green), connective (yellow)

    elif dataset=="pannuke":
        print(f"Dataset in visualize: {dataset}")
        # BGR values used in color
        #colors = [[0  ,   0,   255], [255  ,   200, 0], [0, 255,   0], 
                #[0, 255, 255], [255, 0,   255], [127, 127,   127], [255  ,   255, 0]]
        # color order nuclei: nolable (red), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (cyan)
    
        colors = [[0  ,   0,   255], [255  ,   200, 0], [0, 255,   0], 
                [0, 255, 255], [127, 127,   127], [255, 0,   255]]
         # color order nuclei: nolable (red), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (pink)

    os.makedirs(f"{output_dir}/images/", exist_ok=True)
    
    for img_idx, (img, mask_gt, mask_pred) in enumerate(zip(imgs, ann, pred)):
        overlay_gt, pixel_count = draw_dilation(img, mask_gt[:, :, 0], mask_gt[:, :, 1], colors, nuclei_marker, dataset)
        overlay_pred, pixel_count = draw_dilation(img, mask_pred[:, :, 0], mask_pred[:, :, 1], colors, nuclei_marker, dataset)
        img_to_write = np.zeros((mask_gt.shape[0], 2*mask_gt.shape[1]+5, 3))
        img_to_write[:, :mask_gt.shape[1], :] = overlay_gt
        img_to_write[:, -mask_gt.shape[1]:, :] = overlay_pred
        img_to_write = img_to_write.astype(np.uint8)  # Convert to uint8
        output_path = f"{output_dir}/images/{img_idx}_overlay.png"
        cv2.imwrite(output_path, img_to_write)
    # Convert from BGR to RGB if necessary (OpenCV uses BGR by default)
    # img_to_write_rgb = cv2.cvtColor(img_to_write, cv2.COLOR_BGR2RGB)

    # # Create a figure and axis
    # fig, ax = plt.subplots()

    # # Display the image
    # ax.imshow(img_to_write_rgb)
    # ax.axis('off')  # Hide axis
    # plt.show()

def visualize_no_gt(imgs, imgs_names, pred, output_dir, dataset, nuclei_marker):
    # Decide colors for visualization of nuclei based on datset used
    if dataset=="conic":
        print(f"Dataset in visualize: {dataset}")
        # BGR values used in color
        # colors = [[0  ,   0,   255], [0,   0,   0], [255  ,   255, 0], 
        #         [255 ,   0, 0], [255, 0,   255], [0, 255,   0], [0, 255, 255]]
        # # color order nuclei: nolable (red), neutrophil (black), epithelial (cyan), lymphocyte (dark blue), plasma (magenta), eosinophil (green), connective (yellow)
        colors = [[0,   0,   0], [255  ,   255, 0], 
                [255 ,   0, 0], [255, 0,   255], [0, 255,   0], [0, 255, 255]]
        # color order nuclei: neutrophil (black), epithelial (cyan), lymphocyte (dark blue), plasma (magenta), eosinophil (green), connective (yellow)

    elif dataset=="pannuke":
        print(f"Dataset in visualize: {dataset}")
        # BGR values used in color
        # colors = [[0  ,   0,   255], [255  ,   200, 0], [0, 255,   0], 
        #         [0, 255, 255], [255, 0,   255], [127, 127,   127], [255  ,   255, 0]]
        # # color order nuclei: nolable (red), neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (cyan)
        colors = [[255  ,   200, 0], [0, 255,   0], 
                [0, 255, 255], [127, 127,   127], [255, 0,   255]]
        # color order nuclei: neoplastic (light blue), inflammatory (green), connective (yellow), dead (grey?), epithelial (pink)
    
    os.makedirs(output_dir, exist_ok=True)
    pixel_count_list = []
    for img_idx, (img, mask_pred) in enumerate(zip(imgs, pred)):
        overlay_pred, pixel_count = draw_dilation(img, mask_pred[:, :, 0], mask_pred[:, :, 1], colors, nuclei_marker, dataset)
        pixel_count_list.append(pixel_count)
        img_to_write = overlay_pred
        img_to_write[:, -mask_pred.shape[1]:, :] = overlay_pred
        img_to_write = img_to_write.astype(np.uint8)  # Convert to uint8
        output_path = f"{output_dir}/overlay_{imgs_names[img_idx]}"
        cv2.imwrite(output_path, img_to_write)
    # Convert from BGR to RGB if necessary (OpenCV uses BGR by default)
    # img_to_write_rgb = cv2.cvtColor(img_to_write, cv2.COLOR_BGR2RGB)

    # # Create a figure and axis
    # fig, ax = plt.subplots()

    # # Display the image
    # ax.imshow(img_to_write_rgb)
    # ax.axis('off')  # Hide axis
    # plt.show()

    pixel_count_df = pd.DataFrame(pixel_count_list).fillna(0).astype(int)
    return pixel_count_df

def setup(rank, world_size, port='12353'):
    """
    :param rank: rank of current thread
    :param world_size: the total number of threads
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    clean up the opened threads
    """
    dist.destroy_process_group()


def run_function(func, world_size):
    """
    :param func: the function to call in each thread
    :param world_size: the world size of the distributed system
    """
    mp.spawn(func,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array

    Returns:
        x: cropped array

    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
    return x

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

####
def gen_instance_hv_map(ann, crop_shape):
    """Input annotation must be of original shape.
    
    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = ann.copy()  # instance ID map
    crop_ann = orig_ann.copy()
    # TODO: deal with 1 label warning

    crop_ann = morph.remove_small_objects(crop_ann, min_size=8)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(crop_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue
        
        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        if math.isnan(inst_com[0]):
            continue
        
        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map])
    return hv_map


def gen_instance_hv_slash_map(ann, crop_shape):
    """Input annotation must be of original shape.
    
    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = ann.copy()  # instance ID map
    crop_ann = orig_ann.copy()
    # TODO: deal with 1 label warning

    crop_ann = morph.remove_small_objects(crop_ann, min_size=8)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    slash_45_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    slash_135_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(crop_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)
        
        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        inst_45 = inst_y + inst_x
        inst_135 = inst_y - inst_x

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        if np.min(inst_45) < 0:
            inst_45[inst_45 < 0] /= -np.amin(inst_45[inst_45 < 0])
        if np.min(inst_135) < 0:
            inst_135[inst_135 < 0] /= -np.amin(inst_135[inst_135 < 0] )

        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])
        if np.max(inst_45) > 0:
            inst_45[inst_45 > 0] /= np.amax(inst_45[inst_45 > 0])
        if np.max(inst_135) > 0:
            inst_135[inst_135 > 0] /= np.amax(inst_135[inst_135 > 0])

        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        slash_45_box = slash_45_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        slash_45_box[inst_map > 0] = inst_45[inst_map > 0]

        slash_135_box = slash_135_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        slash_135_box[inst_map > 0] = inst_135[inst_map > 0]

    hv_map = np.dstack([x_map, y_map, slash_45_map, slash_135_map])
    return hv_map


def gen_instance_eight_axixes_map(ann, crop_shape):
    """Input annotation must be of original shape.
    
    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = ann.copy()  # instance ID map
    crop_ann = orig_ann.copy()
    # TODO: deal with 1 label warning

    crop_ann = morph.remove_small_objects(crop_ann, min_size=8)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    slash_45_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    slash_135_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    x_map_neg = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map_neg = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    slash_45_map_neg = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    slash_135_map_neg = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(crop_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        if math.isnan(inst_com[0]):
            continue

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)
        
        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        inst_45 = inst_y + inst_x
        inst_135 = inst_y - inst_x

        # normalize min into -1 scale
        inst_x_neg, inst_y_neg = inst_x.copy(), inst_y.copy()
        inst_45_neg, inst_135_neg = inst_45.copy(), inst_135.copy()
        inst_x_neg, inst_y_neg = -inst_x_neg, -inst_y_neg
        inst_45_neg, inst_135_neg = -inst_45_neg, -inst_135_neg

        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] = 0
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] = 0
        if np.min(inst_45) < 0:
            inst_45[inst_45 < 0] = 0
        if np.min(inst_135) < 0:
            inst_135[inst_135 < 0] = 0

        if np.min(inst_x_neg) < 0:
            inst_x_neg[inst_x_neg < 0] = 0
        if np.min(inst_y_neg) < 0:
            inst_y_neg[inst_y_neg < 0] = 0
        if np.min(inst_45_neg) < 0:
            inst_45_neg[inst_45_neg < 0] = 0
        if np.min(inst_135_neg) < 0:
            inst_135_neg[inst_135_neg < 0] = 0
        

        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])
        if np.max(inst_45) > 0:
            inst_45[inst_45 > 0] /= np.amax(inst_45[inst_45 > 0])
        if np.max(inst_135) > 0:
            inst_135[inst_135 > 0] /= np.amax(inst_135[inst_135 > 0])

        if np.max(inst_x_neg) > 0:
            inst_x_neg[inst_x_neg > 0] /= np.amax(inst_x_neg[inst_x_neg > 0])
        if np.max(inst_y_neg) > 0:
            inst_y_neg[inst_y_neg > 0] /= np.amax(inst_y_neg[inst_y_neg > 0])
        if np.max(inst_45_neg) > 0:
            inst_45_neg[inst_45_neg > 0] /= np.amax(inst_45_neg[inst_45_neg > 0])
        if np.max(inst_135_neg) > 0:
            inst_135_neg[inst_135_neg > 0] /= np.amax(inst_135_neg[inst_135_neg > 0])

        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        slash_45_box = slash_45_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        slash_45_box[inst_map > 0] = inst_45[inst_map > 0]

        slash_135_box = slash_135_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        slash_135_box[inst_map > 0] = inst_135[inst_map > 0]

        x_map_box_neg = x_map_neg[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box_neg[inst_map > 0] = inst_x_neg[inst_map > 0]

        y_map_box_neg = y_map_neg[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box_neg[inst_map > 0] = inst_y_neg[inst_map > 0]

        slash_45_box_neg = slash_45_map_neg[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        slash_45_box_neg[inst_map > 0] = inst_45_neg[inst_map > 0]

        slash_135_box_neg = slash_135_map_neg[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        slash_135_box_neg[inst_map > 0] = inst_135_neg[inst_map > 0]

    hv_map = np.dstack([x_map, y_map, slash_45_map, slash_135_map, x_map_neg, y_map_neg, slash_45_map_neg, slash_135_map_neg])
    return hv_map


def gen_targets_hv_slash(ann, crop_shape):
    """Generate the targets for the network."""
    hv_map = gen_instance_hv_slash_map(ann, crop_shape)
    np_map = ann.copy()
    np_map[np_map > 0] = 1

    target_dict = {
        "hv_map": hv_map,
        "np_map": np_map,
    }

    return target_dict


def gen_targets_hv_eight_axis(ann, crop_shape):
    """Generate the targets for the network."""
    hv_map = gen_instance_eight_axixes_map(ann, crop_shape)
    np_map = ann.copy()
    np_map[np_map > 0] = 1

    target_dict = {
        "hv_map": hv_map,
        "np_map": np_map,
    }

    return target_dict

####
def gen_targets(ann, crop_shape):
    """Generate the targets for the network."""
    hv_map = gen_instance_hv_map(ann, crop_shape)
    np_map = ann.copy()
    np_map[np_map > 0] = 1

    target_dict = {
        "hv_map": hv_map,
        "np_map": np_map,
    }

    return target_dict