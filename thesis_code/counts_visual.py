#Visualize counts from csv

import pandas as pd
import matplotlib.pyplot as plt
import csv

#path to csv file
#20x
#best model
csv_04320_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func043_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_04420_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func044_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_05020_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func050_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_305.csv'
csv_11620_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func116_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_271.csv'

#second model
csv_04320_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func043_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_04420_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func044_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_05020_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func050_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_305.csv'
csv_11620_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func116_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_271.csv'


#40x
#best model
csv_04340_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func043_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_04440_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func044_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_05040_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func050_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_11640_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'

#second model
csv_04340_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func043_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_04440_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func044_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_05040_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func050_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_11640_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'


#Read the csv file
df = pd.read_csv(csv_04440_second)
df.head()

#Sum up counts for each cell type
totals = df[['neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial']].sum()
print(totals)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].bar(totals.index, totals.values)
axes[0].set_ylabel('Cell count')
axes[0].set_title('Total cell counts per type')
axes[0].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()