import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

#For comparing the two models 
#File path to excel file
#file_path = "/Volumes/Expansion/supervisors/Fra Vilde 09.12/SpatialdataModels.xlsx" Pannuke model 1 vs 2

file_path = "/Users/tirilkt/Documents/studie/masteroppgave/spatial-data/conic/Figures/Func116SpotQuantification1.xlsx" #Pannuke vs conic

#Read file
df = pd.read_excel(file_path)

#Select paired data
#Drop rows that has NaN

#Pannuke model 1 vs 2
# neo_pair = df[['HoverNet model1 Neoplastic', 'HoverNet model2 Neoplastic']]
# con_pair = df[['HoverNet model1 Connective', 'HoverNet model2 Connective']]
# inf_pair = df[['HoverNet model1 Inflammatory', 'HoverNet model2 Inflammatory']]

# model1_neo = neo_pair['HoverNet model1 Neoplastic']
# model2_neo = neo_pair['HoverNet model2 Neoplastic']
# model1_con = con_pair['HoverNet model1 Connective']
# model2_con = con_pair['HoverNet model2 Connective']
# model1_inf = inf_pair['HoverNet model1 Inflammatory']
# model2_inf = inf_pair['HoverNet model2 Inflammatory']

#Pannuke vs conic model
neo_pair = df[['PanNuke model2 Neoplastic', 'Conic 20x Epithelial']]
con_pair = df[['PanNuke model2 Connective', 'Conic 20x Connective']]
#inf_pair = df[['HoverNet model1 Inflammatory', 'HoverNet model2 Inflammatory']]

model1_neo = neo_pair['PanNuke model2 Neoplastic']
model2_neo = neo_pair['Conic 20x Epithelial']
model1_con = con_pair['PanNuke model2 Connective']
model2_con = con_pair['Conic 20x Connective']
#model1_inf = inf_pair['HoverNet model1 Inflammatory']
#model2_inf = inf_pair['HoverNet model2 Inflammatory']

def pearson(x_name, y_name):
    pair = df[[x_name, y_name]].dropna()   #Remove NaN values
    corr, p = pearsonr(pair[x_name], pair[y_name])
    print(f"Pearson correlation (r) {x_name} vs {y_name}: {corr:.3f}")
    print(f"Pearson p-value ({x_name} vs {y_name}): {p:.3e}")
    return corr, p

# pearson("HoverNet model1 Neoplastic", "HoverNet model2 Neoplastic")
# pearson("HoverNet model1 Connective", "HoverNet model2 Connective") 
# pearson("HoverNet model1 Inflammatory", "HoverNet model2 Inflammatory") 

pearson("PanNuke model2 Neoplastic", "Conic 20x Epithelial") #Pearson correlation HoverNet Conic Epithelial vs SpaCET Malignant: 
pearson("PanNuke model2 Connective", "Conic 20x Connective") #Pearson correlation Hovernet Conic Connective vs SpaCET CAF: 
#pearson("Conic 40x Lymphocyte", "SpaCET Immune_cells") #Pearson correlation HoverNet Conic Lymphocite vs SpaCET Immune_cells: 


#Plots

# plt.figure(figsize=(8, 8))
# sns.regplot(x=model1_neo, y=model2_neo, scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"}
# )

# plt.title("HoVer-Net Neoplastic - Model 1 vs Model 2 ")
# plt.xlabel("Model 1 Neoplastic")
# plt.ylabel("Model 2 Neoplastic")
# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(8, 8))
# sns.regplot(x=model1_con, y=model2_con, scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"}
# )

# plt.title("HoVer-Net Connective - Model 1 vs Model 2 ")
# plt.xlabel("Model 1 Connective")
# plt.ylabel("Model 2 Connective")
# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 8))
# sns.regplot(x=model1_inf, y=model2_inf, scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"}
# )

# plt.title("HoVer-Net Inflammatory - Model 1 vs Model 2 ")
# plt.xlabel("Model 1 Inflammatory")
# plt.ylabel("Model 2 Inflammatory")
# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


plt.figure(figsize=(8, 8))
sns.regplot(x=model1_neo, y=model2_neo, scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"}
)

plt.title("PanNuke Neoplastic - CoNIC Epithelial")
plt.xlabel("PanNuke Neoplastic")
plt.ylabel("CoNIC Epithelial")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 8))
sns.regplot(x=model1_con, y=model2_con, scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"}
)

plt.title("PanNuke Connective - CoNIC Connective ")
plt.xlabel("PanNuke Connective")
plt.ylabel("CoNIC Connective")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()