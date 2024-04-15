import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
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


fig = plt.figure(figsize=(80, 60))
# fig.patch.set_facecolor('white')
Z = linkage(np.array(data_2020_scale), method='single')
R = dendrogram(Z, labels=list(subjects), orientation = 'left', leaf_font_size = 8)

# plt.axvline(x=11, color='r', linestyle='--')
# Отображение дендрограммы
plt.show()