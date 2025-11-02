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

def eval_models(imgs_load_array, image_names, tp_num, exp_name0, encoder_name0, exp_name1, encoder_name1, output_dir, output_dir_dataframe, output_dir_dataframe_p, patch_mask_binary, checkpoint0, checkpoint1, epoch_idx=79, dataset="pannuke", nuclei_marker="fill"):
    valid_indices = range(len(imgs_load_array))
    #valid_indices = range(len(imgs_load_array-1 ))
    
    # Checkpoint path for seresnext50
    checkpoint_path0 = os.path.join(checkpoint0, f"improved-net_{epoch_idx}.pt")
    segmentation_model0 = HoVerNetHeadExt(num_types=tp_num, encoder_name=encoder_name0, pretrained_backbone=None)
    #checkpoint_path0 = f"{args.checkpoint0}/improved-net_{epoch_idx}.pt"
    #segmentation_model0 = HoVerNetHeadExt(num_types, encoder_name=encoder_name0, pretrained_backbone=None)
    
    # Checkpoint path for seresnext101
    checkpoint_path1 = os.path.join(checkpoint1, f"improved-net_{epoch_idx}.pt")
    segmentation_model1 = HoVerNetHeadExt(num_types=tp_num, encoder_name=encoder_name1, pretrained_backbone=None)
    #checkpoint_path1 = f"{args.checkpoint1}/improved-net_{epoch_idx}.pt"
    #segmentation_model1 = HoVerNetHeadExt(num_types, encoder_name=encoder_name1, pretrained_backbone=None)
    
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
    
        # Apply mask to the patch (set pixels in tissue fold to white)
        img[patch_mask_binary.squeeze() == 0] = [255, 255, 255]
       
        # Assuming imgs_valid contains the images in the desired format:
        # Here we normalize the image and prepare for PyTorch tensor conversion
        img = img[None, :, :, :] / 255.  # Normalize the image to [0, 1]
        
        #img_tensor = torch.tensor(img)   
        img_tensor = torch.from_numpy(img).float().to(0)

        # print(f"Shape after unsqueeze: {img_tensor.shape}")
        np_map0, hv_map0, tp_map0 = segmentation_model0.infer_batch_inner_ensemble(segmentation_model0, img_tensor, True, idx=idx, encoder_name="seresnext50")
        np_map1, hv_map1, tp_map1 = segmentation_model1.infer_batch_inner_ensemble(segmentation_model1, img_tensor, True, idx=idx, encoder_name="seresnext101")

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
    labels_array_pred, nuclei_counts_df_pred, nuclei_counts_array_pred = prepare_results(np_results, hv_results, tp_results, segmentation_model0, patch_shape=[img_tensor.shape[1],img_tensor.shape[2]], tp_num = tp_num, dataset = dataset)
    
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
    pixel_count_df.to_csv(output_dir_dataframe_p, index=False)

    
def read_images(tile_path: str, start:int, end:int):
    """
    Load all images and output them as a sorted array.
    """
    image_list = glob.glob(tile_path + '*.png')     # Finds all images with .png extension
    image_list = [p for p in image_list if not os.path.basename(p).startswith('._')] 
    image_list = natsorted(image_list)              # Sorts the images naturally

    #Take a section of the patches
    original_len = len(image_list)

    if end == -1 or end > len(image_list):
        end = len(image_list)
    image_list = image_list[start:end]

    print(f"Total patches: {original_len}. Loading {start}:{end} and {len(image_list)} patches")

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

def read_mask(tile_path: str, start:int, end:int):
    """
    Load all masks and output them as a sorted array.
    """
    mask_list = glob.glob(tile_path + '*.png')
    mask_list = [p for p in mask_list if not os.path.basename(p).startswith('._')]
    mask_list = natsorted(mask_list)

    #image_list = glob.glob(tile_path + '*.png')     # Finds all images with .png extension
    #image_list = natsorted(image_list)              # Sorts the images naturally

    #Take a section of the mask
    if end == -1 or end > len(mask_list):
        end = len(mask_list)
    mask_list = mask_list[start:end]
    
    imgs_load_list = []
    # Loop through all masks
    for i in range(len(mask_list)):
        image_fetch = mask_list[i]
        image = cv2.imread(image_fetch)             # Load the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # imgs_load = np.array(image)  
        imgs_binary = (image != 255).astype(np.uint8)  # Black pixels set to 1 and white pixels set to 0
        imgs_binary = imgs_binary[:, :, np.newaxis]  # Apply on each channel
        imgs_load_list.append(imgs_binary)
    mask_load_array = np.array(imgs_load_list)
    print(f"Loaded array of masks with shape: {mask_load_array.shape}")
    return mask_load_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    #parser.add_argument("--dataset", choose=["pannuke", "conic"], required=True)

    parser.add_argument("--model", type=str, default="hovernet")
    parser.add_argument("--log_name", type=str, default="ensemble_all_fold_0_pannuke")
    
    parser.add_argument('--exp_name0', type=str, default='hover_paper_pannuke_seresnext50')
    #parser.add_argument('--exp_name0', type=str, default='hover_paper_conic_seresnext50_00')
    parser.add_argument("--encoder_name0", type=str, default="seresnext50")
    
    parser.add_argument('--exp_name1', type=str, default='hover_paper_pannuke_seresnext101')
    #parser.add_argument('--exp_name1', type=str, default='hover_paper_conic_seresnext101_00')
    parser.add_argument("--encoder_name1", type=str, default="seresnext101")
    
    parser.add_argument("--nuclei_marker", choices = ["border", "fill"], default="border", help ="Choose how you want nuclei to be marked in overlay. Choose either 'border' or 'fill' (default: %(default)s)" )
    # nuclei_marker = "fill"    # If you want the whole nuclei colored
    # nuclei_marker = "border"  # If you only want the nuclei border/outline marked
    
    parser.add_argument("--tile_path", type=str, required=True, help = "Path to images you want analyzed")
    # tile_path = "/.../patches/HE_xxx/"
    parser.add_argument("--mask_path", type=str, required=True, help = "Path to binary masks") 
    # mask_path = "/.../masks/HE_xxx/" 
    parser.add_argument("--output_dir_dataframe", type=Path, required=True, help = "Output directory for dataframe") 
    # output_dir_dataframe = Path("/.../Output/patches/HE_xxx/counts/")
    parser.add_argument("--output_dir_dataframe_p", type=Path, required=True, help = "Output directory for dataframe counting number of pixels for each type of nucleus")  
    # output_dir_dataframe_p = Path("/.../Output/patches/HE_xxx/count_pixels/")
    parser.add_argument("--output_dir", type=str, required=True, help = "Output directory for overlay images")  
    # output_dir = "/.../Output/patches/HE_xxx/overlay/"

    # parser argument for path til checkpoint fil
    parser.add_argument("--checkpoint0", type=str, required=True, help="Path til checkpoint fil for model 0")
    parser.add_argument("--checkpoint1", type=str, required=True, help="Path til checkpoint fil for model 1")

    #Take out a section of the patches from start to end (to reduce memory)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1) #-1 means the last element, so to the end
    
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
    epoch_idx = 79
    
    tile_path = args.tile_path
    # Extract array of images and image names
    imgs_load_array, image_names = read_images(tile_path, args.start, args.end)
    
    # Extract array of masks
    patch_mask_binary = read_mask(args.mask_path, args.start, args.end)

    # Extracting numbers using string manipulation
    numbers = [int(name.split('_')[1].split('.')[0]) for name in image_names]
    # Create directory if it doesn't exist
    if not args.output_dir_dataframe.exists(): 
        args.output_dir_dataframe.mkdir(parents=True)
        print(f"Directory {args.output_dir_dataframe} was created")
    output_dir_dataframe = os.path.join(args.output_dir_dataframe, f"nuclei_counts_from_{numbers[0]}_to_{numbers[-1]}.csv")
    
    # Create directory if it doesn't exist
    if not args.output_dir_dataframe_p.exists(): 
        args.output_dir_dataframe_p.mkdir(parents=True)
        print(f"Directory {args.output_dir_dataframe_p} was created")
    output_dir_dataframe_p = os.path.join(args.output_dir_dataframe_p, f"pixel_count_from_{numbers[0]}_to_{numbers[-1]}.csv")

    # Output directory for overlay images (will be automatically created when used as input in visualize_no_gt() if it doesn't already exist)

    eval_models(imgs_load_array, image_names, num_types, \
                args.exp_name0, args.encoder_name0, \
                args.exp_name1, args.encoder_name1, \
                checkpoint0=args.checkpoint0, checkpoint1=args.checkpoint1, \
                epoch_idx=epoch_idx, dataset=dataset, output_dir=args.output_dir, output_dir_dataframe=output_dir_dataframe, output_dir_dataframe_p = output_dir_dataframe_p, nuclei_marker=args.nuclei_marker, patch_mask_binary = patch_mask_binary)

        