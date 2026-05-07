import pandas as pd

"""
Combining two CSV files that contain the cell counts for the 40x samples from inference.
"""
#CSV file paths
csv1 = '/Volumes/Expansion/biopsy_results/conic/40x/output_fill/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_from_0_to_599.csv'
csv2 = '/Volumes/Expansion/biopsy_results/conic/40x/output_fill/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_from_600_to_1087.csv'

df1 = pd.read_csv(csv1) #Load CSV files
df2 = pd.read_csv(csv2)
combined = pd.concat([df1, df2], ignore_index=True) #Combine CSV files

#Save as CSV file
combined.to_csv('/Volumes/Expansion/biopsy_results/conic/40x/output_fill/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv', index=False)