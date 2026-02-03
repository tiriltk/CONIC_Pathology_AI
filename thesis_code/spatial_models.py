import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

#File path to excel spatial data from supervisor
#file_path = "/Volumes/Expansion/supervisors/Fra Vilde 09.12/SpatialdataModels.xlsx"
file_path = "/Users/tirilkt/Documents/studie/masteroppgave/spatial-data/conic/Figures/Func116SpotQuantification1.xlsx"

#Read file
df = pd.read_excel(file_path)

# Pannuke
# neoplastic_data = df['HoverNet model1 Neoplastic']
# malignant_data = df['SpaCET Malignant']

# connective_data = df['HoverNet model1 Connective']
# caf_data = df['SpaCET CAF']

# inflammatory_data = df['HoverNet model1 Inflammatory']
# immune_data = df['SpaCET Immune_cells']

# Conic
neoplastic_data = df['Conic 40x Epithelial']
malignant_data = df['SpaCET Malignant']

connective_data = df['Conic 40x Connective']
caf_data = df['SpaCET CAF']

inflammatory_data = df['Conic 40x Lymphocyte']
immune_data = df['SpaCET Immune_cells']

def pearson(x_name, y_name):
    pair = df[[x_name, y_name]].dropna()   #Remove NaN values
    corr, p = pearsonr(pair[x_name], pair[y_name])
    print(f"Pearson correlation (r) {x_name} vs {y_name}: {corr:.3f}")
    print(f"Pearson p-value ({x_name} vs {y_name}): {p:.3e}")
    return corr, p

#Pannuke
# pearson("HoverNet model1 Neoplastic", "SpaCET Malignant") #Pearson correlation HoverNet Neoplastic vs SpaCET Malignant: 0.739
# pearson("HoverNet model1 Connective", "SpaCET CAF") #Pearson correlation HoverNet Connective vs SpaCET CAF: 0.640
# pearson("HoverNet model1 Inflammatory", "SpaCET Immune_cells") #Pearson correlation HoverNet Inflammatory vs SpaCET Immune_cells: 0.160

#Conic
pearson("Conic 40x Epithelial", "SpaCET Malignant")
pearson("Conic 40x Connective", "SpaCET CAF")
pearson("Conic 40x Lymphocyte", "SpaCET Immune_cells")

#Pannuke
# plt.figure(figsize=(8, 8))
# pair = df[["HoverNet model1 Neoplastic", "SpaCET Malignant"]].dropna()
# sns.regplot(x = pair["HoverNet model1 Neoplastic"], y = pair["SpaCET Malignant"], scatter_kws = {"s": 10, "alpha": 0.6}, line_kws = {"color": "black"})

#Pannuke
# plt.title(f"HoVer-Net Neoplastic vs SpaCET Malignant")
# plt.xlabel("HoVer-Net Neoplastic")
# plt.ylabel("SpaCET Malignant")
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#Conic
plt.figure(figsize=(8, 8))
pair = df[["Conic 40x Epithelial", "SpaCET Malignant"]].dropna()
sns.regplot(x = pair["Conic 40x Epithelial"], y = pair["SpaCET Malignant"], scatter_kws = {"s": 10, "alpha": 0.6}, line_kws = {"color": "black"})

#Conic
plt.title(f"HoVer-Net Epithelial vs SpaCET Malignant")
plt.xlabel("HoVer-Net Epithelial")
plt.ylabel("SpaCET Malignant")
plt.xlim(-0.05, 1.05)  
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

#Pannuke
# plt.figure(figsize=(8,8))
# pair = df[["HoverNet model1 Connective", "SpaCET CAF"]].dropna()
# sns.regplot(x = pair["HoverNet model1 Connective"], y = pair["SpaCET CAF"], scatter_kws = {"s":10, "alpha":0.6}, line_kws = {"color":"black"})

# plt.title(f"HoVer-Net Connective vs SpaCET CAF")
# plt.xlabel("HoVer-Net Connective")
# plt.ylabel("SpaCET CAF")
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#Conic
plt.figure(figsize=(8,8))
pair = df[["Conic 40x Connective", "SpaCET CAF"]].dropna()
sns.regplot(x = pair["Conic 40x Connective"], y = pair["SpaCET CAF"], scatter_kws = {"s":10, "alpha":0.6}, line_kws = {"color":"black"})

plt.title(f"HoVer-Net Connective vs SpaCET CAF")
plt.xlabel("HoVer-Net Connective")
plt.ylabel("SpaCET CAF")
plt.xlim(-0.05, 1.05)  
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()


#Pannuke
# plt.figure(figsize=(8,8))
# pair = df[["HoverNet model1 Inflammatory", "SpaCET Immune_cells"]].dropna()
# sns.regplot(x = pair["HoverNet model1 Inflammatory"], y = pair["SpaCET Immune_cells"], scatter_kws = {"s":10, "alpha":0.6}, line_kws = {"color":"black"})

# plt.title(f"HoVer-Net Inflammatory vs SpaCET Immune Cells")
# plt.xlabel("HoVer-Net Inflammatory")
# plt.ylabel("SpaCET Immune Cells")
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#Conic
plt.figure(figsize=(8,8))
pair = df[["Conic 40x Lymphocyte", "SpaCET Immune_cells"]].dropna()
sns.regplot(x = pair["Conic 40x Lymphocyte"], y = pair["SpaCET Immune_cells"], scatter_kws = {"s":10, "alpha":0.6}, line_kws = {"color":"black"})

plt.title(f"HoVer-Net Lymphocyte vs SpaCET Immune Cells")
plt.xlabel("HoVer-Net Lymphocyte")
plt.ylabel("SpaCET Immune Cells")
plt.xlim(-0.05, 1.05)  
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()


#For Pannuke Model 2
#File path to excel spatial data from supervisor
# file_path = "/Volumes/Expansion/supervisors/Fra Vilde 09.12/SpatialdataModels.xlsx"

# #Read file
# df = pd.read_excel(file_path)

# neoplastic_data = df['HoverNet model2 Neoplastic']
# malignant_data = df['SpaCET Malignant']

# connective_data = df['HoverNet model2 Connective']
# caf_data = df['SpaCET CAF']

# inflammatory_data = df['HoverNet model2 Inflammatory']
# immune_data = df['SpaCET Immune_cells']

# def pearson(x_name, y_name):
#     pair = df[[x_name, y_name]].dropna()   #Remove NaN values
#     corr, p = pearsonr(pair[x_name], pair[y_name])
#     print(f"Pearson correlation (r) {x_name} vs {y_name}: {corr:.3f}")
#     print(f"Pearson p-value ({x_name} vs {y_name}): {p:.3e}")
#     return corr, p

# pearson("HoverNet model2 Neoplastic", "SpaCET Malignant") 
# pearson("HoverNet model2 Connective", "SpaCET CAF") 
# pearson("HoverNet model2 Inflammatory", "SpaCET Immune_cells") 

# plt.figure(figsize=(8, 8))
# pair = df[["HoverNet model2 Neoplastic", "SpaCET Malignant"]].dropna()
# sns.regplot(x = pair["HoverNet model2 Neoplastic"], y = pair["SpaCET Malignant"], scatter_kws = {"s": 10, "alpha": 0.6}, line_kws = {"color": "black"})

# plt.title(f"HoVer-Net Neoplastic vs SpaCET Malignant")
# plt.xlabel("HoVer-Net Neoplastic")
# plt.ylabel("SpaCET Malignant")
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,8))
# pair = df[["HoverNet model2 Connective", "SpaCET CAF"]].dropna()
# sns.regplot(x = pair["HoverNet model2 Connective"], y = pair["SpaCET CAF"], scatter_kws = {"s":10, "alpha":0.6}, line_kws = {"color":"black"})

# plt.title(f"HoVer-Net Connective vs SpaCET CAF")
# plt.xlabel("HoVer-Net Connective")
# plt.ylabel("SpaCET CAF")
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,8))
# pair = df[["HoverNet model2 Inflammatory", "SpaCET Immune_cells"]].dropna()
# sns.regplot(x = pair["HoverNet model2 Inflammatory"], y = pair["SpaCET Immune_cells"], scatter_kws = {"s":10, "alpha":0.6}, line_kws = {"color":"black"})

# plt.title(f"HoVer-Net Inflammatory vs SpaCET Immune Cells")
# plt.xlabel("HoVer-Net Inflammatory")
# plt.ylabel("SpaCET Immune_cells")
# plt.xlim(-0.05, 1.05)  
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


