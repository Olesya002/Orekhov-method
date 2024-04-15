import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

'''Будем использовать kmeans для кластеризации временных рядов'''
# from tslearn.generators import random_walks
# X = random_walks(n_ts=50, sz=32, d=2)
# print(X)


data_2020 = pd.read_excel('D:\ВКР\main_data.xlsx', sheet_name='2020', usecols=list(range(0,7)))
data_2020.columns = ['subject', 'VRP', 'INVEST', 'FUNDS', 'COEFF', 'RESEARCH', 'PRODUCT']

data_2019 = pd.read_excel('D:\ВКР\main_data.xlsx', sheet_name='2019', skiprows=[22,23] +[63,64,65], usecols=list(range(0,7)))
data_2019.columns = ['subject', 'VRP', 'INVEST', 'FUNDS', 'COEFF', 'RESEARCH', 'PRODUCT']

# функция для регионов с 2018 по 2014
def data_preprocess(list_name:str):
  data = pd.read_excel('D:\ВКР\main_data.xlsx', sheet_name=list_name, skiprows=[1] + [20] + [24,25] + [33] + [42] + [50] + [65] + [69,70,71] + [73] + [84], usecols=list(range(0,7)))
  data.columns = ['subject', 'VRP', 'INVEST', 'FUNDS', 'COEFF', 'RESEARCH', 'PRODUCT']
  return pd.DataFrame(data)

data_2018 = data_preprocess('2018')
data_2017 = data_preprocess('2017')
data_2016 = data_preprocess('2016')
data_2015 = data_preprocess('2015')
data_2014 = data_preprocess('2014')
data_2013 = pd.read_excel('D:\ВКР\main_data.xlsx', sheet_name='2013', skiprows=[1] + [20] + [24,25] + [33] + [36] + [41, 42] + [50] + [65] + [69,70,71] + [73] + [84], usecols=list(range(0,7)))
data_2012 = pd.read_excel('D:\ВКР\main_data.xlsx', sheet_name='2012', skiprows=[1] + [20] + [24,25] + [33] + [36] + [41, 42] + [50] + [65] + [69,70,71] + [73] + [84], usecols=list(range(0,7)))
data_2012.columns = ['subject', 'VRP', 'INVEST', 'FUNDS', 'COEFF', 'RESEARCH', 'PRODUCT']
data_2011 = pd.read_excel('D:\ВКР\main_data.xlsx', sheet_name='2011', skiprows=[1] + [20] + [24,25] + [33] + [36] + [41, 42] + [50] + [65] + [69,70,71] + [73] + [84], usecols=list(range(0,7)))
data_2013.columns = ['subject', 'VRP', 'INVEST', 'FUNDS', 'COEFF', 'RESEARCH', 'PRODUCT']
data_2011.columns = ['subject', 'VRP', 'INVEST', 'FUNDS', 'COEFF', 'RESEARCH', 'PRODUCT']

'''Сохраним список регионов'''
subjects = data_2020['subject']
subjects_dict = dict(zip(range(82), subjects))
data_2020 = data_2020.drop(['subject'], axis=1)
data_2019 = data_2019.drop(['subject'], axis=1)
data_2018 = data_2018.drop(['subject'], axis=1)
data_2017 = data_2017.drop(['subject'], axis=1)
data_2016 = data_2016.drop(['subject'], axis=1)
data_2015 = data_2015.drop(['subject'], axis=1)
data_2014 = data_2014.drop(['subject'], axis=1)
data_2013 = data_2013.drop(['subject'], axis=1)
data_2012 = data_2012.drop(['subject'], axis=1)
data_2011 = data_2011.drop(['subject'], axis=1)
'''Стандартизация'''
def scaled(df):
  return (df - df.mean())/df.std()
data_2020_scale = scaled(data_2020)

data_2019_scale = scaled(data_2019)
data_2018_scale = scaled(data_2018)
data_2017_scale = scaled(data_2017)
data_2016_scale = scaled(data_2016)
data_2015_scale = scaled(data_2015)
data_2014_scale = scaled(data_2014)
data_2013_scale = scaled(data_2013)
data_2012_scale = scaled(data_2012)
data_2011_scale = scaled(data_2011)

'''Применяем PCA для сокращения признакового пространства'''
pca = PCA(n_components=2)
pca.fit(data_2020_scale)
resulted = pca.transform(data_2020_scale)
print(resulted)

kmeans = KMeans(n_clusters=4)
kmeans.fit(resulted)
print(kmeans.labels_)

'''функция для определения размеров кластеров'''
def coun(nums):
  dict_nums = {}
  for num in nums:
    dict_nums[num] = dict_nums.get(num, 0) + 1
  return dict_nums

'''функция для определения самого частого разбиения из n'''
def find_frequent(n, df):
  list_of_models = []
  list_of_sizes = []
  for i in range(n):
    kmeans = KMeans(n_clusters=4, max_iter=20, n_init=5, init='random').fit(df)
    list_of_models.append(kmeans)
    list_of_sizes.append(tuple(sorted(list(coun(kmeans.labels_).values()))))
  dict_of_frequent = coun(list_of_sizes)
  return list_of_models, dict_of_frequent

models_2020, my_dict = find_frequent(40, resulted)
print(coun(models_2020[2].labels_))
print(subjects[models_2020[2].labels_ == 0])

'''Попробуем разделить датасет на две части'''
economic_data_1 = data_2020_scale.iloc[:, :3]
invest_data_1 = data_2020_scale.iloc[:, 3:]
economic_data_2 = data_2019_scale.iloc[:, :3]
invest_data_2 = data_2019_scale.iloc[:, 3:]
'''И к каждой из них применим PCA'''
pca1 = PCA(n_components=1)
pca1.fit(economic_data_1)
economic_1_feature = pca1.transform(economic_data_1)
pca1.fit(economic_data_2)
economic_2_feature = pca1.transform(economic_data_2)
print(economic_1_feature)
pca2 = PCA(n_components=1)
pca2.fit(invest_data_1)
invest_1_feature = pca2.transform(invest_data_1)
pca2.fit(invest_data_2)
invest_2_feature = pca2.transform(invest_data_2)
print(invest_1_feature)

data_2020_2_features = np.hstack((economic_1_feature, invest_1_feature))
data_2019_2_features = np.hstack((economic_2_feature, invest_2_feature))
timeseries_1 = np.array([[[0., 0.] for i in range(2)] for j in range(len(subjects))])
print(timeseries_1)
for i in range(len(subjects)):
  timeseries_1[i] = [data_2020_2_features[i], data_2019_2_features[i]]
print(timeseries_1)
kmeans = TimeSeriesKMeans(n_clusters=4, metric='euclidean', max_iter=20)
kmeans.fit(timeseries_1)
print(coun(kmeans.labels_))
print(subjects[kmeans.labels_ == 1])
print(subjects[kmeans.labels_ == 2])
print(subjects[kmeans.labels_ == 0])
print(subjects[kmeans.labels_ == 3])
'''Результат отличается от ожидаемого разбиения'''


'''Временные ряды с использованием всех признаков'''
timeseries_2 = np.array([[[0., 0., 0., 0., 0., 0.] for i in range(2)] for j in range(len(subjects))])
for i in range(len(subjects)):
  timeseries_2[i] = [np.array(data_2020_scale)[i], np.array(data_2019_scale)[i]]
print(timeseries_2)
kmeans = TimeSeriesKMeans(n_clusters=4, metric='euclidean', max_iter=20)
kmeans.fit(timeseries_2)
print(coun(kmeans.labels_))
print(subjects[kmeans.labels_ == 1])
print(subjects[kmeans.labels_ == 2])
print(subjects[kmeans.labels_ == 0])
print(subjects[kmeans.labels_ == 3])