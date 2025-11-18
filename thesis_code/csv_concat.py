#Concatenating CSV
#combining two csv with same structure

import pandas as pd

#CSV file paths
csv1 = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_from_0_to_599.csv'
csv2 = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_from_600_to_1087.csv'

#Load the CSV files into DataFrames
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

#Concentate the DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

#Save combined DataFrame to new csv file
#change path here to the correct biopsy folder!
combined_df.to_csv('/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv', index=False)

print("combined csv is made!")