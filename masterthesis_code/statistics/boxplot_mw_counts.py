import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

#Box Plots and Mann Whitney Test when comparing Pannuke Model 1 and Model 2
#Using cell counts from hovernet results
#Same classes and same lengths (20x and 20x or 40x and 40x)

#Path to csv file that contains the counts for each patch
#Best model performance metrics (Model 1) 20x
csv_04320_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func043_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_04420_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func044_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_05020_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func050_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_305.csv'
csv_11620_best = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_best/Func116_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_271.csv'

#Second model performance metrics (Model 2) 20x
csv_04320_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func043_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_04420_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func044_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_323.csv'
csv_05020_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func050_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_305.csv'
csv_11620_second = '/Volumes/Expansion/biopsy_results/pannuke/20x/datafiles_output_20x_second/Func116_ST_HE_20x_BF_01/counts/nuclei_counts_from_0_to_271.csv'

#Best model performance metrics (Model 1) 40x
csv_04340_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func043_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_04440_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func044_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_05040_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func050_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_11640_best = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_best/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'

#Second model performance metrics (Model 2) 40x
csv_04340_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second_old/Func043_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_04440_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second_old/Func044_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_05040_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second_old/Func050_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'
csv_11640_second = '/Volumes/Expansion/biopsy_results/pannuke/40x/datafiles_output_40x_second_old/Func116_ST_HE_40x_BF_01/counts/nuclei_counts_combined.csv'

#Pannuke cell types
pannuke_types = ['neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial']

#Unfiltered contains patches with zero counts because there are no cells in some patches
model1_unfiltered =  pd.read_csv(csv_11620_best) 
model2_unfiltered = pd.read_csv(csv_11620_second)

plot_data = []

for cell_type in pannuke_types:
    #Mask to keep patches where at least one of the models finds cells
    #Filter out rows where both are 0
    mask = (model1_unfiltered[cell_type] > 0) | (model2_unfiltered[cell_type] > 0) #logical OR operator
    model1_filtered = model1_unfiltered[mask]
    model2_filtered = model2_unfiltered[mask]

    #Mann Whitney U Test
    stat, p = mannwhitneyu(model1_filtered[cell_type], model2_filtered[cell_type], alternative='two-sided')
    print(f"{cell_type}: U Statistics = {stat:.2f}, P Value = {p:.4f}")

    plot_data.append(pd.DataFrame({"Value": model1_filtered[cell_type].values, "Cell type": cell_type, "Model": "Model 1"}))
    plot_data.append(pd.DataFrame({"Value": model2_filtered[cell_type].values, "Cell type": cell_type, "Model": "Model 2"}))

df_visualize = pd.concat(plot_data, ignore_index=True)

#Boxplot
plt.figure(figsize = (10,6))
sns.boxplot(x = 'Cell type', y = 'Value', hue = 'Model', data = df_visualize, palette = ['royalblue', 'hotpink'])
plt.title('Comparison of cell counts between Model 1 and Model 2')
plt.ylabel('Cell counts per patch')
plt.xlabel("Cell type")
plt.grid(axis = 'y', linestyle = '--', alpha=0.5)
plt.legend(title = 'Model')
plt.tight_layout()
plt.show()

#Histograms
for cell_type in pannuke_types:
    mask = (model1_unfiltered[cell_type] > 0) | (model2_unfiltered[cell_type] > 0)
    model1_filtered = model1_unfiltered[mask]
    model2_filtered = model2_unfiltered[mask]
    plt.figure(figsize = (6,4))
    plt.hist(model1_filtered[cell_type], bins = 20, alpha = 0.6, label = 'Model 1', color = 'royalblue')
    plt.hist(model2_filtered[cell_type], bins = 20, alpha = 0.6, label = 'Model 2', color = 'hotpink')
    plt.title(f"Histogram of {cell_type} counts")
    plt.xlabel("Counts")
    plt.ylabel("Patches")
    plt.legend()
    plt.tight_layout()
    plt.show()