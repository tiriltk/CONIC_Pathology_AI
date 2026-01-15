import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

#File path to excel 
file_path = "/Volumes/Expansion/supervisors/Fra Vilde 09.12/SpatialdataModels.xlsx"

#Read file
df = pd.read_excel(file_path)

neoplastic_data = df['HoverNet model2 Neoplastic']
malignant_data = df['SpaCET Malignant']

connective_data = df['HoverNet model2 Connective']
caf_data = df['SpaCET CAF']

inflammatory_data = df['HoverNet model2 Inflammatory']
immune_data = df['SpaCET Immune_cells']

def pearson(x_name, y_name):
    pair = df[[x_name, y_name]].dropna()   #Remove NaN 
    corr, p = pearsonr(pair[x_name], pair[y_name])
    print(f"Pearson correlation (r) {x_name} vs {y_name}: {corr:.3f}")
    print(f"Pearson p-value ({x_name} vs {y_name}): {p:.3e}")
    return corr, p

pearson("HoverNet model2 Neoplastic", "SpaCET Malignant") 
pearson("HoverNet model2 Connective", "SpaCET CAF") 
pearson("HoverNet model2 Inflammatory", "SpaCET Immune_cells") 

plt.figure(figsize=(8, 8))
pair = df[["HoverNet model2 Neoplastic", "SpaCET Malignant"]].dropna()
sns.regplot(x=pair["HoverNet model2 Neoplastic"], y=pair["SpaCET Malignant"], scatter_kws={"s": 10, "alpha": 0.6}, line_kws={"color": "black"})

plt.title(f"HoVer-Net Neoplastic vs SpaCET Malignant")
plt.xlabel("HoVer-Net Neoplastic")
plt.ylabel("SpaCET Malignant")
plt.xlim(-0.05, 1.05)  
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,8))
pair = df[["HoverNet model2 Connective", "SpaCET CAF"]].dropna()
sns.regplot(x=pair["HoverNet model2 Connective"], y=pair["SpaCET CAF"], scatter_kws={"s":10, "alpha":0.6}, line_kws={"color":"black"})

plt.title(f"HoVer-Net Connective vs SpaCET CAF")
plt.xlabel("HoVer-Net Connective")
plt.ylabel("SpaCET CAF")
plt.xlim(-0.05, 1.05)  
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,8))
pair = df[["HoverNet model2 Inflammatory", "SpaCET Immune_cells"]].dropna()
sns.regplot(x=pair["HoverNet model2 Inflammatory"], y=pair["SpaCET Immune_cells"], scatter_kws={"s":10, "alpha":0.6}, line_kws={"color":"black"})

plt.title(f"HoVer-Net Inflammatory vs SpaCET Immune Cells")
plt.xlabel("HoVer-Net Inflammatory")
plt.ylabel("SpaCET Immune_cells")
plt.xlim(-0.05, 1.05)  
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()


