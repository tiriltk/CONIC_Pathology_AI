from PIL import Image
import os
import argparse

"""
Divide downscaled WSIs into four parts to reduce memory usage.
"""

def dividing_wsi(wsi_path, output_dir): #Function to divide image into four parts and save them
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) #Create output directory

    wsi = Image.open(wsi_path) #Open image file
    width, height = wsi.size #Width and height of the image
    half_width = width // 2 #Calculate half the width and height to find the center of the image
    half_height = height // 2

    #Coordinates for each part using half width and half height and using crop to extract the parts
    topleft_wsi = wsi.crop((0, 0, half_width, half_height)) #top left part
    save_path = os.path.join(output_dir, "top_left.png")
    topleft_wsi.save(save_path) #save cropped image

    topright_wsi = wsi.crop((half_width, 0, width, half_height)) #top right part
    save_path = os.path.join(output_dir, "top_right.png")
    topright_wsi.save(save_path) #save cropped image

    bottomleft_wsi = wsi.crop((0, half_height, half_width, height)) #bottom left part
    save_path = os.path.join(output_dir, "bottom_left.png")
    bottomleft_wsi.save(save_path) #save cropped image

    bottomright_wsi = wsi.crop((half_width, half_height, width, height)) #bottom right part
    save_path = os.path.join(output_dir, "bottom_right.png")
    bottomright_wsi.save(save_path) #save cropped image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide WSIs into parts") #Parser arguments
    parser.add_argument("input_path", type=str, help="Path to the input WSI file")
    parser.add_argument("--output", dest="output_dir", type=str, help="Output directory for the divided parts")
    args = parser.parse_args()
    dividing_wsi(args.input_path, args.output_dir)
