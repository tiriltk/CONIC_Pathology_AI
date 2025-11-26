#Mann-Whitney U test
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

#Path to csv file
#20x
#Best model
csv_04320_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func043_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_04420_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func044_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_05020_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func050_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_305.csv'
csv_11620_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func116_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_271.csv'

#Second model
csv_04320_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func043_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_04420_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func044_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_05020_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func050_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_305.csv'
csv_11620_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func116_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_271.csv'

#40x
#Best model
csv_04340_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func043_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_04440_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func044_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_05040_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func050_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_11640_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'

#Second model
csv_04340_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func043_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_04440_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func044_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_05040_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func050_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_11640_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'

#Pannuke cell types
cell_types = ['neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial']

#Model_unfiltered contains patches with zero counts because there are no cells in some areas
model1_unfiltered =  pd.read_csv(csv_11640_best) 
model2_unfiltered = pd.read_csv(csv_11640_second)

#Mask to keep patches where at least one of the models finds cells
#Conditions
condition1 = (model1_unfiltered[cell_types].sum(axis=1) > 0) #calculates sum for each row
condition2 = (model2_unfiltered[cell_types].sum(axis=1) > 0)
mask = condition1 | condition2

#Masks on both unfiltered models
model1 = model1_unfiltered[mask]
model2 = model2_unfiltered[mask]

print(model1)
print(model2)

assert len(model1_unfiltered) == len(model2_unfiltered), "Mismatch in patch count"
assert len(model1) == len(model2), "Mismatch after masking"


#Mann-Whitney U Test
#Print
p_values = {}
print("Mann–Whitney U test results:")
for t in cell_types:
    stat, p = mannwhitneyu(model1[t], model2[t], alternative='two-sided')
    p_values[t] = p
    print(f"{t}: U Statistics = {stat:.2f}, P Value = {p:.4f}")

#Plotting to visualize
df_visualize = pd.DataFrame({
    'Value': pd.concat([model1[cell_types].melt()['value'], model2[cell_types].melt()['value']]),
    'Cell type': list(model1[cell_types].melt()['variable'])*2,
    'Model': ['Model 1']*len(model1[cell_types].melt()) + ['Model 2']*len(model2[cell_types].melt())})

#Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x='Cell type', y='Value', hue='Model', data=df_visualize, palette=['royalblue', 'hotpink'])
plt.title('Comparison of cell counts between Model 1 and 2')
plt.ylabel('Cell counts per patch')
plt.xlabel("Cell type")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title='Model')
plt.tight_layout()
plt.show()


#Histograms
for type in cell_types:
    plt.figure(figsize=(6,4))
    plt.hist(model1[type], bins=20, alpha=0.6, label='Model 1', color='royalblue')
    plt.hist(model2[type], bins=20, alpha=0.6, label='Model 2', color='hotpink')
    plt.title(f"Histogram of {type} counts")
    plt.xlabel("Counts")
    plt.ylabel("Patches")
    plt.legend()
    plt.tight_layout()
    plt.show()

