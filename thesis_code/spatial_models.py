import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#File path to excel 
file_path = "/Volumes/Expansion/supervisors/Fra Vilde 09.12/SpatialdataModels.xlsx"

#Read file
df = pd.read_excel(file_path)

#Select paired data
#Drop rows that has NaN
neo_pair = df[['HoverNet model1 Neoplastic', 'HoverNet model2 Neoplastic']].dropna()
con_pair = df[['HoverNet model1 Connective', 'HoverNet model2 Connective']].dropna()
inf_pair = df[['HoverNet model1 Inflammatory', 'HoverNet model2 Inflammatory']].dropna()

model1_neo = neo_pair['HoverNet model1 Neoplastic']
model2_neo = neo_pair['HoverNet model2 Neoplastic']
model1_con = con_pair['HoverNet model1 Connective']
model2_con = con_pair['HoverNet model2 Connective']
model1_inf = inf_pair['HoverNet model1 Inflammatory']
model2_inf = inf_pair['HoverNet model2 Inflammatory']

plt.figure(figsize=(8, 8))
sns.regplot(x=model1_neo, y=model2_neo, scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"}
)

plt.title("HoVer-Net Neoplastic - Model 1 vs Model 2 ")
plt.xlabel("Model 1 Neoplastic")
plt.ylabel("Model 2 Neoplastic")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 8))
sns.regplot(x=model1_con, y=model2_con, scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"}
)

plt.title("HoVer-Net Connective - Model 1 vs Model 2 ")
plt.xlabel("Model 1 Connective")
plt.ylabel("Model 2 Connectve")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 8))
sns.regplot(x=model1_inf, y=model2_inf, scatter_kws={"s": 10, "alpha": 0.6, "color": "navy"}, line_kws={"color": "black"}
)

plt.title("HoVer-Net Inflammatory - Model 1 vs Model 2 ")
plt.xlabel("Model 1 Inflammatory")
plt.ylabel("Model 2 Inflammatory")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()