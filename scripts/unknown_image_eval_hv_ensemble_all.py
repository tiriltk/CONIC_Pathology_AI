import os
import sys
sys.path.append('./')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import cv2
import joblib
import argparse
from itertools import islice
import warnings
warnings.filterwarnings("ignore")
import glob
import torch
from natsort import natsorted
from models.model_head_aug import HoVerNetHeadExt
from utils.eval_utils import prepare_ground_truth, prepare_results, convert_pytorch_checkpoint
from utils.util_funcs import visualize_no_gt

from PIL import Image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval_models(imgs_load_array, image_names, tp_num, exp_name0, encoder_name0, exp_name1, encoder_name1, output_dir, output_dir_dataframe, output_dir_dataframe_p,patch_mask_binary, epoch_idx=49, dataset="conic", nuclei_marker="fill"):

    valid_indices = range(len(imgs_load_array-1 ))
    
    # Checkpoint path for seresnext50
    # checkpoint_path0 = f"/media/jenny/PRIVATE_USB/AugHoverData/pannuke_checkpoints_trained/checkpoints/batch_size_8/{exp_name0}/improved-net_{epoch_idx}.pt"
    checkpoint_path0 = f"/media/jenny/Expansion/Conic_weights/checkpoints/{exp_name0}/improved-net_{epoch_idx}.pt"
    segmentation_model0 = HoVerNetHeadExt(num_types, encoder_name=encoder_name0, pretrained_backbone=None)
    
    # Checkpoint path for seresnext101
    # checkpoint_path1 = f"/media/jenny/PRIVATE_USB/AugHoverData/pannuke_checkpoints_trained/checkpoints/batch_size_8/{exp_name1}/improved-net_{epoch_idx}.pt"
    checkpoint_path1 = f"/media/jenny/Expansion/Conic_weights/checkpoints/{exp_name1}/improved-net_{epoch_idx}.pt"
    segmentation_model1 = HoVerNetHeadExt(num_types, encoder_name=encoder_name1, pretrained_backbone=None)
    
    print(f"===================parameter counts: {count_parameters(segmentation_model0) + count_parameters(segmentation_model1)}=====")
    print(f"===================parameter counts: {count_parameters(segmentation_model0)} ")
    print(f"===================parameter counts: {count_parameters(segmentation_model1)} ")
    state_dict = torch.load(checkpoint_path0)        # Load checkpoints
    segmentation_model0.load_state_dict(state_dict)
    segmentation_model0 = segmentation_model0.to(0)
    segmentation_model0.eval()                       # Set module in evaluation mode
    
    state_dict = torch.load(checkpoint_path1)        # Load checkpoints
    segmentation_model1.load_state_dict(state_dict)
    segmentation_model1 = segmentation_model1.to(0)
    segmentation_model1.eval()                       # Set module in evaluation mode
    np_results, hv_results, tp_results = [], [], []
    imgs_valid = imgs_load_array[valid_indices] 

    print(f"valid_indices: {valid_indices}")
    
    for idx, img, patch_mask_binary in tqdm(zip(valid_indices, imgs_valid, patch_mask_binary), total=len(valid_indices)):
    
        # Apply mask to the patch (e.g., zero out excluded pixels)
        # masked_patch = img * patch_mask_binary
        masked_patch = img
       
        # Assuming imgs_valid contains the images in the desired format:
        # Here we normalize the image and prepare for PyTorch tensor conversion
        masked_patch = masked_patch[None, :, :, :] / 255.  # Normalize the image to [0, 1]
        
        img_tensor = torch.tensor(masked_patch)  
    
        # img_tensor = img_tensor.unsqueeze(0)  # This adds a batch dimension
        # print(f"Shape after unsqueeze: {img_tensor.shape}")
        np_map0, hv_map0, tp_map0 = segmentation_model0.infer_batch_inner_ensemble(segmentation_model0, img_tensor, True, idx=idx, encoder_name="seresnext50")
        np_map1, hv_map1, tp_map1 = segmentation_model1.infer_batch_inner_ensemble(segmentation_model1, img_tensor, True, idx=idx, encoder_name="seresnext101")
        
        
        for name, module in segmentation_model0.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):  # For 2D BatchNorm layers
                print(f"{name} running_mean: {module.running_mean}")
                print(f"{name} running_var: {module.running_var}")
        
        np_map = (np_map0[0] + np_map1[0]) / 2
        hv_map = (hv_map0[0] + hv_map1[0]) / 2
        tp_map = (tp_map0[0] + tp_map1[0]) / 2

        tp_map = np.argmax(tp_map, axis=-1)
        tp_map = np.array(tp_map, np.float32)
        tp_map = tp_map[:, :, None]

        np_results.append(np_map)
        hv_results.append(hv_map)
        tp_results.append(tp_map)

    print(f"img_tensor.shape[0]: {img_tensor.shape[0]}")
    print(f"img_tensor.shape[1]: {img_tensor.shape[1]}")
    print(f"img_tensor.shape[2]: {img_tensor.shape[2]}")
    print(f"img_tensor.shape[3]: {img_tensor.shape[3]}")
    
    # Extract results
    labels_array_pred, nuclei_counts_df_pred, nuclei_counts_array_pred = prepare_results(np_results, hv_results, tp_results, segmentation_model0, patch_shape=[img_tensor.shape[1],img_tensor.shape[2]])
    
    # Insert column with patch number for each result
    column_name = "patch_nbr"
    column_values = [int(name.split('_')[1].split('.')[0]) for name in image_names] # Extracting image numbers using string manipulation
    nuclei_counts_df_pred.insert(0, column_name, column_values)
    # Print and save the dataframe with the nuclei types and counts
    print(f'{nuclei_counts_df_pred}')
    nuclei_counts_df_pred.to_csv(output_dir_dataframe, index=False)
    
    # Create and save overlay + extract dataframe with pixel counts for nuclei types
    pixel_count_df = visualize_no_gt(imgs=imgs_load_array, imgs_names=image_names, pred=labels_array_pred , output_dir=output_dir, dataset=dataset, nuclei_marker=nuclei_marker)
    pixel_count_df.insert(0, column_name, column_values)
    # Print and save the dataframe
    print(pixel_count_df)
    pixel_count_df.to_csv(output_dir_dataframe_p)


def read_images(tile_path: str | Path):
    """
    Load all images and output them as a sorted array.
    """

    image_list = glob.glob(tile_path + '*.png')     # Finds all images with .png extension
    image_list = natsorted(image_list)              # Sorts the images naturally
    # Extract only the image names from the paths
    image_names = [os.path.basename(image) for image in image_list]
    
    imgs_load_list = []
    # Loop through all images
    for i in range(len(image_list)):
        image_fetch = image_list[i]
        image = Image.open(image_fetch)             # Load the image
        imgs_load = np.array(image)                 # Convert the image to a NumPy array
        imgs_load = np.array(image.convert("RGB"))  # If image is RGBA or other formats, convert it to RGB
        imgs_load_list.append(imgs_load)
    imgs_load_array = np.array(imgs_load_list)
    print(f"Loaded array of images with shape: {imgs_load_array.shape}")
    
    return imgs_load_array, image_names

def read_mask(tile_path: str | Path):
    """
    Load all masks and output them as a sorted array.
    """

    image_list = glob.glob(tile_path + '*.png')     # Finds all images with .png extension
    image_list = natsorted(image_list)              # Sorts the images naturally
    
    imgs_load_list = []
    # Loop through all masks
    for i in range(len(image_list)):
        image_fetch = image_list[i]
        image = cv2.imread(image_fetch)             # Load the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # imgs_load = np.array(image)  
        imgs_binary = (image != 255).astype(np.uint8) 
        imgs_binary = imgs_binary[:, :, np.newaxis]  # Apply on each channel
        print(f"imgs_mask shape: {imgs_binary.shape}") # Convert the image to a NumPy array
        imgs_load_list.append(imgs_binary)
    mask_load_array = np.array(imgs_load_list)
    print(f"Loaded array of masks with shape: {imgs_load_array.shape}")
    
    return mask_load_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--model", type=str, default="hovernet")
    parser.add_argument("--log_name", type=str, default="ensemble_all_fold_0_conic")
    
    # parser.add_argument('--exp_name0', type=str, default='hover_paper_pannuke_seresnext50')
    parser.add_argument('--exp_name0', type=str, default='hover_paper_conic_seresnext50_00')
    parser.add_argument("--encoder_name0", type=str, default="seresnext50")
    
    # parser.add_argument('--exp_name1', type=str, default='hover_paper_pannuke_seresnext101')
    parser.add_argument('--exp_name1', type=str, default='hover_paper_conic_seresnext101_00')
    parser.add_argument("--encoder_name1", type=str, default="seresnext101")

    args = parser.parse_args()
    
    # Ensures correct parameters for chosen weights 
    if "pannuke" in args.exp_name0:
        num_types = 6
        dataset = "pannuke"
        print(f"dataset: {dataset}")
    elif "conic" in args.exp_name0:
        num_types = 7
        dataset = "conic"
        print(f"Dataset used: {dataset}")

    # Choose epoch you want to retrieve weights from
    epoch_idx = 49
    
    # Choose how you want nuclei to be marked in overlay
    nuclei_marker = "fill"    # If you want the whole nuclei colored
    # nuclei_marker = "border"  # If you only want the nuclei border/outline marked
    
    # Path to images you want analyzed
    tile_path = "/media/jenny/Expansion/MM_HE_patches/HE_MM009_B_270125/aughovernet/extra_patch/" 
    # Extract array of images and image names
    imgs_load_array, image_names = read_images(tile_path)
    
    # Path to binary masks 
    mask_path = "/media/jenny/Expansion/MM_HE_patches/HE_MM009_B_270125/aughovernet/extra/" 
    # Extract array of masks
    patch_mask_binary = read_mask(mask_path)

    # Output directory for dataframe counting number of nuclei
    output_dir_dataframe = Path("/media/jenny/Expansion/MM_HE_patches/HE_MM009_B_270125/aughovernet/debug_change_no_mask/tiles_result_csv_8/")
    if not output_dir_dataframe.exists(): 
        output_dir_dataframe.mkdir(parents=True)
        print(f"Directory {output_dir_dataframe} was created")
    output_dir_dataframe = os.path.join(output_dir_dataframe, f"nuclei_counts.csv")
    
    # Output directory for dataframe counting number of pixels for each type of nucleus 
    output_dir_dataframe_p = Path("/media/jenny/Expansion/MM_HE_patches/HE_MM009_B_270125/aughovernet/debug_change/tiles_result_pixels_csv_8/")
    if not output_dir_dataframe_p.exists(): 
        output_dir_dataframe_p.mkdir(parents=True)
        print(f"Directory {output_dir_dataframe_p} was created")
    output_dir_dataframe_p = os.path.join(output_dir_dataframe_p, f"pixel_count.csv")
    
    # Output directory for overlay images (will be automatically created when used as input in visualize_no_gt() if it doesn't already exist)
    output_dir = "/media/jenny/Expansion/MM_HE_patches/HE_MM009_B_270125/aughovernet/debug_change/tiles_result_8/"
    
    eval_models(imgs_load_array, image_names, num_types, \
                args.exp_name0, args.encoder_name0, \
                args.exp_name1, args.encoder_name1, \
                epoch_idx=epoch_idx, dataset=dataset, output_dir=output_dir, output_dir_dataframe=output_dir_dataframe, output_dir_dataframe_p = output_dir_dataframe_p, nuclei_marker=nuclei_marker, patch_mask_binary = patch_mask_binary)
    
    