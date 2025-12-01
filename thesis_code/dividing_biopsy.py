from PIL import Image
import os
import sys
import argparse

"""
Divides the PNG image into four parts and saves them to be able to open the biopsies
"""

def split_image(image_path, output_dir):
    img = Image.open(image_path)

    width, height = img.size

    #Calculate half the width and height
    half_width = width // 2
    half_height = height // 2

    #Coordinates for each part (left, top, right, bottom)
    topleft = (0, 0, half_width, half_height) #top left
    topright = (half_width, 0, width, half_height) #top right 
    bottomleft = (0, half_height, half_width, height) #bottom left
    bottomright = (half_width, half_height, width, height) #bottom right

    parts = [("top_left.png", topleft), ("top_right.png", topright), ("bottom_left.png", bottomleft), ("bottom_right.png", bottomright)]

    #Create a directory to save the output images if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for name, part in parts:
        cropped_img = img.crop(part)
        save_path = os.path.join(output_dir, name)
        cropped_img.save(save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split PNG image into equal parts")
    parser.add_argument('input_image', type=str, help='Path to the input PNG file')
    parser.add_argument('--output', dest='output_dir', type=str, help='The destination directory for the output images')

    args = parser.parse_args()

split_image(args.input_image, args.output_dir)


"""
python /Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func116_ST_HE_40x_BF_01/wsi_tp_results/whole_image_complete_scaled.png \
--output /Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func116_ST_HE_40x_BF_01/wsi_tp_results/
"""