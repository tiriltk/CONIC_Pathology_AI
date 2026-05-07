import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

"""
Box plots and Mann Whitney test comparing the selected Pannuke and Conic model using estimates from spot quantification file.
"""

#File path to spot quantification file
file_path = "/Users/tirilkt/Documents/studie/masteroppgave/spatial-data/conic/Figures/Func116SpotQuantification1.xlsx" #Pannuke vs conic
df = pd.read_excel(file_path) #Read file

#Comparing neoplastic, epithelial and connective cell types
pannuke_neo = df['PanNuke model2 Neoplastic'].dropna() 
pannuke_con = df['PanNuke model2 Connective'].dropna() 
conic_epi = df['Conic 20x Epithelial'].dropna()
conic_con = df['Conic 20x Connective'].dropna() 

#Mann Whitney test
stat_con, p_con = mannwhitneyu(pannuke_con, conic_con, alternative='two-sided') 
stat_mal, p_mal = mannwhitneyu(pannuke_neo, conic_epi, alternative='two-sided')
print(f"PanNuke connective and CoNIC connective: U Statistics = {stat_con:.2f}, P Value = {p_con:.4f}") 
print(f"PanNuke neoplastic and CoNIC epithelial: U Statistics = {stat_mal:.2f}, P Value = {p_mal:.4f}") 

#Plotting 
df_visualize = pd.concat([
    pd.DataFrame({'Value': pannuke_neo, 'Model': 'PanNuke', 'Cell type': 'Malignant'}),
    pd.DataFrame({'Value': conic_epi, 'Model': 'CoNIC', 'Cell type': 'Malignant'}),
    pd.DataFrame({'Value': pannuke_con, 'Model': 'PanNuke', 'Cell type': 'Connective'}),
    pd.DataFrame({'Value': conic_con, 'Model': 'CoNIC', 'Cell type': 'Connective'})
])

#Boxplot
sns.boxplot(data = df_visualize, x = 'Cell type', y = 'Value', hue = 'Model', palette=['hotpink', 'royalblue'])
plt.title('Comparison of cell estimates between CoNIC and PanNuke models')
plt.ylabel('Cell fraction per spatial spot')
plt.xlabel('')
plt.grid(axis = 'y', linestyle = '--', alpha = 0.5)
plt.legend(title='Model')
plt.tight_layout()
plt.show()

# plt.figure(figsize = (6,4))
# plt.hist(pannuke_neo, bins = 20, alpha = 0.6, label = 'Model 1', color = 'royalblue')
# plt.hist(conic_epi, bins = 20, alpha = 0.6, label = 'Model 2', color ='hotpink')
# plt.title(f"Histogram of cell fractions")
# plt.xlabel("Estimates")
# plt.ylabel("Epithelial/Neoplastic")
# plt.legend()
# plt.tight_layout()
# plt.show()


