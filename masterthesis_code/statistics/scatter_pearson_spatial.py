import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

"""
Pearson correlation and scatter plots comparing cell estimates.
Using spot quantification file with model estimates and SpaCET data.
"""

#File paths to spot quantification file
#file_path = "/Volumes/Expansion/supervisors/FraVilde09.12/SpatialdataModels.xlsx" #Pannuke Model 1 and 2
file_path = "/Users/tirilkt/Documents/studie/masteroppgave/spatial-data/conic/Figures/Func116SpotQuantification1.xlsx" #Pannuke and conic
df = pd.read_excel(file_path) #Read file

def pearson(x_data, y_data): #Function for pearson correlation analysis
    estimates = df[[x_data, y_data]].dropna() #Remove NaN values
    pearson_corr, p_value = pearsonr(estimates[x_data], estimates[y_data])
    print(f"Pearson correlation (r) {x_data} and {y_data}: {pearson_corr:.3f}")
    print(f"Pearson p-value ({x_data} and {y_data}): {p_value:.3e}")

#PanNuke and CoNIC
pearson("PanNuke model2 Neoplastic", "Conic 20x Epithelial")
pearson("PanNuke model2 Connective", "Conic 20x Connective") 

#Scatter plots
plt.figure(figsize=(8, 8))
estimates = df[["PanNuke model2 Neoplastic", "Conic 20x Epithelial"]].dropna()
sns.regplot(x=estimates["PanNuke model2 Neoplastic"], y=estimates["Conic 20x Epithelial"], scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"})
plt.title("PanNuke Model 2 (40x) Neoplastic - CoNIC (20x) Epithelial", fontsize=16)
plt.xlabel("PanNuke Model 2 (40x) Neoplastic", fontsize=14)
plt.ylabel("CoNIC (20x) Epithelial", fontsize=14)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
estimates = df[["PanNuke model2 Connective", "Conic 20x Connective"]].dropna()
sns.regplot(x=estimates["PanNuke model2 Connective"], y=estimates["Conic 20x Connective"], scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"})
plt.title("PanNuke Model 2 (40x) Connective - CoNIC (20x) Connective", fontsize=16)
plt.xlabel("PanNuke Model 2 (40x) Connective", fontsize=14)
plt.ylabel("CoNIC (20x) Connective", fontsize=14)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()
