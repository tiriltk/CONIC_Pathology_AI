#Combine the results from tp_results_colored
import os
import shutil

#Combined them with offset in tp_results_colored, so then this is not needed!

def combine_png(part1_path: str, part2_path: str, combined_path: str):
    #Make directory
    os.makedirs(combined_path, exist_ok=True)
    
    part1_files = os.listdir(part1_path)
    part1 = sorted(part1_files)

    part2_files = os.listdir(part2_path)
    part2 = sorted(part2_files)

    #Offset to correct for file names
    offset = len(part1)

    for count, filename in enumerate(part1):
        source = os.path.join(part1_path, filename)
        dest = os.path.join(combined_path, f"tp_patch_{count}.png")
        shutil.copy(source, dest)

    for count, filename in enumerate (part2):
        source = os.path.join(part2_path, filename)
        dest = os.path.join(combined_path, f"tp_patch_{offset+count}.png")
        shutil.copy(source, dest)

    print(f"Combined {len(part1) + len(part2)} to {combined_path}")

if __name__ == "__main__":
    part1 = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func116_ST_HE_40x_BF_01/tp_results_png_colors_part1"
    part2 = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func116_ST_HE_40x_BF_01/tp_results_png_colors_part2"
    combined = "/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/output_fill/Func116_ST_HE_40x_BF_01/tp_results_png_colors_combined"


combine_png(part1, part2, combined)



