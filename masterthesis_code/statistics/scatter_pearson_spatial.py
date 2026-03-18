import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

"""
Pearson correlation and scatter plots
Comparing spatial cell estimates from model predictions and spatial transcriptomics
Using excel file with containing spatial transcriptomics data and model estimates
"""

#Paths for comparing two PanNuke models or two CoNIC models
#file_path = "/Volumes/Expansion/supervisors/Fra Vilde 09.12/SpatialdataModels.xlsx"
#file_path = "/Users/tirilkt/Documents/studie/masteroppgave/spatial-data/conic/Figures/Func116SpotQuantification1.xlsx"

#Paths for comparing PanNuke and CoNIC models 
file_path = "/Volumes/Expansion/supervisors/Fra Vilde 09.12/SpatialdataModels.xlsx" #Pannuke Model 1 and 2
#file_path = "/Users/tirilkt/Documents/studie/masteroppgave/spatial-data/conic/Figures/Func116SpotQuantification1.xlsx" #Pannuke vs conic

df = pd.read_excel(file_path) #Read file

def pearson(x_data, y_data):
    pair = df[[x_data, y_data]].dropna() #Remove NaN values
    corr, p = pearsonr(pair[x_data], pair[y_data])
    print(f"Pearson correlation (r) {x_data} and {y_data}: {corr:.3f}")
    print(f"Pearson p-value ({x_data} and {y_data}): {p:.3e}")

# #Pearson Correlations
# #PanNuke and CoNIC
# pearson("PanNuke model2 Neoplastic", "Conic 20x Epithelial")
# pearson("PanNuke model2 Connective", "Conic 20x Connective") 

# #Scatter plots
# #PanNuke and CoNIC
# plt.figure(figsize=(8, 8))
# pair = df[["PanNuke model2 Neoplastic", "Conic 20x Epithelial"]].dropna()
# sns.regplot(x=pair["PanNuke model2 Neoplastic"], y=pair["Conic 20x Epithelial"], scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"})
# plt.title("PanNuke Model 2 (40x) Neoplastic - CoNIC (20x) Epithelial", fontsize=16)
# plt.xlabel("PanNuke Model 2 (40x) Neoplastic", fontsize=14)
# plt.ylabel("CoNIC (20x) Epithelial", fontsize=14)
# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 8))
# pair = df[["PanNuke model2 Connective", "Conic 20x Connective"]].dropna()
# sns.regplot(x=pair["PanNuke model2 Connective"], y=pair["Conic 20x Connective"], scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"})
# plt.title("PanNuke Model 2 (40x) Connective - CoNIC (20x) Connective", fontsize=16)
# plt.xlabel("PanNuke Model 2 (40x) Connective", fontsize=14)
# plt.ylabel("CoNIC (20x) Connective", fontsize=14)
# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#Pearson correlation
#PanNuke Model 1 and Model 2
pearson("HoverNet model1 Neoplastic", "HoverNet model2 Neoplastic")
pearson("HoverNet model1 Connective", "HoverNet model2 Connective") 

plt.figure(figsize=(8, 8))
pair = df[["HoverNet model1 Neoplastic", "HoverNet model2 Neoplastic"]].dropna()
sns.regplot(x=pair["HoverNet model1 Neoplastic"], y=pair["HoverNet model2 Neoplastic"], scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"})
plt.title("PanNuke Model 1 - Model 2 (40x) Neoplastic", fontsize=16)
plt.xlabel("PanNuke Model 1 (40x) Neoplastic", fontsize=14)
plt.ylabel("PanNuke Model 2 (40x) Neoplastic", fontsize=14)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
pair = df[["HoverNet model1 Connective", "HoverNet model2 Connective"]].dropna()
sns.regplot(x=pair["HoverNet model1 Connective"], y=pair["HoverNet model2 Connective"], scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"})
plt.title("PanNuke Model 1 - Model 2 (40x) Connective", fontsize=16)
plt.xlabel("PanNuke Model 1 (40x) Connective", fontsize=14)
plt.ylabel("PanNuke Model 2 (40x) Connective", fontsize=14)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()


#Pearson correlations
#Conic
# pearson("Conic 20x Epithelial", "SpaCET Malignant")
# pearson("Conic 20x Connective", "SpaCET CAF")
# pearson("Conic 20x Lymphocyte", "SpaCET Immune_cells")

# #Scatter plots
# #Conic
# plt.figure(figsize=(8, 8))
# pair = df[["Conic 20x Epithelial", "SpaCET Malignant"]].dropna()
# sns.regplot(x = pair["Conic 20x Epithelial"], y = pair["SpaCET Malignant"], scatter_kws = {"s": 10, "alpha": 0.6, "color": "navy"}, line_kws = {"color": "black"})
# plt.title(f"CoNIC 20x Epithelial - Spatial Transcriptomics Malignant", fontsize=16)
# plt.xlabel("CoNIC 20x Epithelial", fontsize=14)
# plt.ylabel("Spatial Transcriptomics Malignant", fontsize=14)
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 8))
# pair = df[["Conic 20x Connective", "SpaCET CAF"]].dropna()
# sns.regplot(x = pair["Conic 20x Connective"], y = pair["SpaCET CAF"], scatter_kws = {"s":10, "alpha":0.6, "color": "navy"}, line_kws = {"color":"black"})
# plt.title(f"CoNIC 20x Connective - Spatial Transcriptomics CAF", fontsize=16)
# plt.xlabel("CoNIC 20x Connective", fontsize=14)
# plt.ylabel("Spatial Transcriptomics CAF", fontsize=14)
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 8))
# pair = df[["Conic 20x Lymphocyte", "SpaCET Immune_cells"]].dropna()
# sns.regplot(x = pair["Conic 20x Lymphocyte"], y = pair["SpaCET Immune_cells"], scatter_kws = {"s":10, "alpha":0.6, "color": "navy"}, line_kws = {"color":"black"})
# plt.title(f"CoNIC 20x Lymphocyte vs Spatial Transcriptomics Immune Cells", fontsize=16)
# plt.xlabel("CoNIC 20x Lymphocyte", fontsize=14)
# plt.ylabel("Spatial Transcriptomics Immune Cells", fontsize=14)
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#Pearson correlations
#Pannuke Model 1
# pearson("HoverNet model1 Neoplastic", "SpaCET Malignant") #Pearson correlation HoverNet Neoplastic vs SpaCET Malignant: 0.739
# pearson("HoverNet model1 Connective", "SpaCET CAF") #Pearson correlation HoverNet Connective vs SpaCET CAF: 0.640
# pearson("HoverNet model1 Inflammatory", "SpaCET Immune_cells") #Pearson correlation HoverNet Inflammatory vs SpaCET Immune_cells: 0.160

# #Scatter plots
# #Pannuke Model 1
# plt.figure(figsize=(8, 8))
# pair = df[["HoverNet model1 Neoplastic", "SpaCET Malignant"]].dropna()
# sns.regplot(x = pair["HoverNet model1 Neoplastic"], y = pair["SpaCET Malignant"], scatter_kws = {"s": 10, "alpha": 0.6, "color": "navy"}, line_kws = {"color": "black"})
# plt.title(f"PanNuke Model 1 Neoplastic - Spatial Transcriptomics Malignant", fontsize=16)
# plt.xlabel("PanNuke Model 1 (40x) Neoplastic", fontsize=14)
# plt.ylabel("Spatial Transcriptomics Malignant", fontsize=14)
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,8))
# pair = df[["HoverNet model1 Connective", "SpaCET CAF"]].dropna()
# sns.regplot(x = pair["HoverNet model1 Connective"], y = pair["SpaCET CAF"], scatter_kws = {"s":10, "alpha":0.6, "color": "navy"}, line_kws = {"color":"black"})
# plt.title(f"PanNuke Model 1 Connective - Spatial Transcriptomics CAF", fontsize=16)
# plt.xlabel("PanNuke Model 1 (40x) Connective", fontsize=14)
# plt.ylabel("Spatial Transcriptomics CAF", fontsize=14)
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,8))
# pair = df[["HoverNet model1 Inflammatory", "SpaCET Immune_cells"]].dropna()
# sns.regplot(x = pair["HoverNet model1 Inflammatory"], y = pair["SpaCET Immune_cells"], scatter_kws = {"s":10, "alpha":0.6, "color": "navy"}, line_kws = {"color":"black"})
# plt.title(f"PanNuke Model 1 Inflammatory - Spatial Transcriptomics Immune Cells", fontsize=16)
# plt.xlabel("PanNuke Model 1 (40x) Inflammatory", fontsize=14)
# plt.ylabel("Spatial Transcriptomics Immune Cells", fontsize=14)
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#Pearson correlations
#Pannuke Model 2
# pearson("HoverNet model2 Neoplastic", "SpaCET Malignant") 
# pearson("HoverNet model2 Connective", "SpaCET CAF") 
# pearson("HoverNet model2 Inflammatory", "SpaCET Immune_cells") 

#Scatter plots
#Pannuke Model 2
# plt.figure(figsize=(8, 8))
# pair = df[["HoverNet model2 Neoplastic", "SpaCET Malignant"]].dropna()
# sns.regplot(x = pair["HoverNet model2 Neoplastic"], y = pair["SpaCET Malignant"], scatter_kws = {"s": 10, "alpha": 0.6, "color": "navy"}, line_kws = {"color": "black"})
# plt.title(f"PanNuke Model 2 Neoplastic - Spatial Transcriptomics Malignant", fontsize=16)
# plt.xlabel("PanNuke Model 2 (40x) Neoplastic", fontsize=14)
# plt.ylabel("Spatial Transcriptomics Malignant", fontsize=14)
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,8))
# pair = df[["HoverNet model2 Connective", "SpaCET CAF"]].dropna()
# sns.regplot(x = pair["HoverNet model2 Connective"], y = pair["SpaCET CAF"], scatter_kws = {"s":10, "alpha":0.6, "color": "navy"}, line_kws = {"color":"black"})

# plt.title(f"PanNuke Model 2 Connective - Spatial Transcriptomics CAF", fontsize=16)
# plt.xlabel("PanNuke Model 2 (40x) Connective", fontsize=14)
# plt.ylabel("Spatial Transcriptomics CAF", fontsize=14)
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,8))
# pair = df[["HoverNet model2 Inflammatory", "SpaCET Immune_cells"]].dropna()
# sns.regplot(x = pair["HoverNet model2 Inflammatory"], y = pair["SpaCET Immune_cells"], scatter_kws = {"s":10, "alpha":0.6, "color": "navy"}, line_kws = {"color":"black"})
# plt.title(f"PanNuke Model 2 Inflammatory vs Spatial Transcriptomics Immune Cells", fontsize=16)
# plt.xlabel("PanNuke Model 2 (40x) Inflammatory", fontsize=14)
# plt.ylabel("Spatial Transcriptomics Immune Cells", fontsize=14)
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()
