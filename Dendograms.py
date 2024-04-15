import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, centroid
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

data_2020 = pd.read_excel('main_data.xlsx', sheet_name='2020', usecols=list(range(0,7)))
data_2020.columns = ['subject', 'VRP', 'INVEST', 'FUNDS', 'COEFF', 'RESEARCH', 'PRODUCT']

# субъекты, 82 штуки
subjects = data_2020['subject']

# словарь с соответствующим номером для каждого субъекта
subjects_dict = dict(zip(range(82), subjects))
data_2020 = data_2020.drop(['subject'], axis=1)
def scaled(df):
  return (df - df.mean())/df.std()
data_2020_scale = scaled(data_2020)


# single
fig = plt.figure(figsize=(80, 60))
#fig.patch.set_facecolor('white')
# truncate_mode='level', p=10,
Z = linkage(np.array(data_2020_scale), method='ward')
R = dendrogram(Z, labels=list(subjects), orientation = 'left', leaf_font_size = 12, above_threshold_color = 'skyblue', color_threshold=8)
#plt.fill("j", "k", 'm',data={"j": [0, 0, 2, 2],"k": [0, 9, 9, 0]})
# plt.axvline(x=11, color='r', linestyle='--')
# Отображение дендрограммы
plt.show()

'''from sklearn.decomposition import PCA
pca = PCA(n_components=2)
components = pca.fit_transform(data_2020_scale)
plt.figure(figsize=(8, 6))
plt.scatter(components[:,0], components[:,1], edgecolors='black', color = 'skyblue')
plt.grid(True, alpha=0.5)
plt.show()'''