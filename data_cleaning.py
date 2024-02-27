import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df =  pd.read_csv('HeartDisease.csv')
print('check')
print(df)
df = df.drop('chd',axis=1)
#print(df.describe().round(2))
#HELLO WORLD


corr_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
