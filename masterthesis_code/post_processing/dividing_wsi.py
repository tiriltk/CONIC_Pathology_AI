from PIL import Image
import os
import argparse

"""
Divides the WSI into four parts to save memory when opening the images
"""

def dividing_wsi(wsi_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wsi = Image.open(wsi_path)
    width, height = wsi.size
    
    #Calculate half the width and height to find center
    half_width = width // 2 
    half_height = height // 2

    # #Coordinates for each part and PIL crop
    # #top left
    # topleft = (0, 0, half_width, half_height)
    # topleft_wsi = wsi.crop(topleft)
    # save_path = os.path.join(output_dir, "top_left.png")
    # topleft_wsi.save(save_path)

    #top right
    topright = (half_width, 0, width, half_height)
    topright_wsi = wsi.crop(topright)
    save_path = os.path.join(output_dir, "top_right.png")
    topright_wsi.save(save_path)

    # #bottom left
    # bottomleft = (0, half_height, half_width, height) 
    # bottomleft_wsi = wsi.crop(bottomleft)
    # save_path = os.path.join(output_dir, "bottom_left.png")
    # bottomleft_wsi.save(save_path)

    # #bottom right
    # bottomright = (half_width, half_height, width, height) 
    # bottomright_wsi = wsi.crop(bottomright)
    # save_path = os.path.join(output_dir, "bottom_right.png")
    # bottomright_wsi.save(save_path)

#Practicing parser arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide WSIs into four parts")
    parser.add_argument("input_path", type=str, help='Path to the input WSI file')
    parser.add_argument("--output", dest="output_dir", type=str, help="Output directory for the four divided parts")
    args = parser.parse_args()
    
    dividing_wsi(args.input_path, args.output_dir)

"""
python /Users/tirilkt/Documents/studie/masteroppgave/CONIC_Pathology_AI/masterthesis_code/post_processing/dividing_wsi.py \
/Volumes/Expansion/biopsy_results/conic/20x/output_fill/Func116_ST_HE_20x_BF_01/wsi_fill/whole_image_scaled.png \
--output /Volumes/Expansion/biopsy_results/conic/20x/output_fill/Func116_ST_HE_20x_BF_01/wsi_fill_testing/
"""
