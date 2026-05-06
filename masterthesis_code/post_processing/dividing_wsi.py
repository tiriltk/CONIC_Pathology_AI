from PIL import Image
import os
import argparse

"""
Divide downscaled WSIs into four parts to reduce memory usage.
"""

def dividing_wsi(wsi_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wsi = Image.open(wsi_path)
    width, height = wsi.size
    half_width = width // 2 #Calculate half the width and height to find the center
    half_height = height // 2

    #Coordinates for each part and using crop to extract parts
    topleft_wsi = wsi.crop((0, 0, half_width, half_height)) #top left
    save_path = os.path.join(output_dir, "top_left.png")
    topleft_wsi.save(save_path)

    topright_wsi = wsi.crop((half_width, 0, width, half_height)) #top right
    save_path = os.path.join(output_dir, "top_right.png")
    topright_wsi.save(save_path)

    bottomleft_wsi = wsi.crop((0, half_height, half_width, height)) #bottom left
    save_path = os.path.join(output_dir, "bottom_left.png")
    bottomleft_wsi.save(save_path)

    bottomright_wsi = wsi.crop((half_width, half_height, width, height)) #bottom right
    save_path = os.path.join(output_dir, "bottom_right.png")
    bottomright_wsi.save(save_path)

#Practicing using parser arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide WSIs into parts")
    parser.add_argument("input_path", type=str, help="Path to the input WSI file")
    parser.add_argument("--output", dest="output_dir", type=str, help="Output directory for the divided parts")
    args = parser.parse_args()

    dividing_wsi(args.input_path, args.output_dir)
