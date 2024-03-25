import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


'''Во втором разделе я исследую метод Орехова и выясняю будет ли он работать на моём примере'''
'''Подгрузим данные за 10 лет'''


data_2020 = pd.read_excel('D:\ВКР\main_data.xlsx', sheet_name='2020', usecols=list(range(0,7)))
data_2020.columns = ['subject', 'VRP', 'INVEST', 'FUNDS', 'COEFF', 'RESEARCH', 'PRODUCT']

'''Сохраним список регионов'''
subjects = data_2020['subject']
subjects_dict = dict(zip(range(82), subjects))
data_2020 = data_2020.drop(['subject'], axis=1)

'''Стандартизация'''
def scaled(df):
  return (df - df.mean())/df.std()
data_2020_scale = scaled(data_2020)
start_df = data_2020_scale.copy()



'''Вычисление погрешностей аппроксимации (линейной и параболической)'''
def delta_l(li):
    k = len(li)
    a = (6/(k*(k**2 - 1))) * sum(list(map(lambda x: (2 * x + 1 - k) * li[x], range(k))))
    b = (2 / (k * (k + 1))) * sum(list(map(lambda x: (2 * k - 1 - 3*x) * li[x], range(k))))
    res = sum(list(map(lambda x: (a * x + b - li[x]) ** 2, range(k))))
    return res

def delta_q(li):
    k = len(li)
    c = 30 / (k * (k - 1) * (2*k - 1) * (8 * k**2 - 3*k - 11)) * sum(list(map(lambda x: (6 * x**2 - (k - 1)*(2*k - 1))*li[x], range(k))))
    d = 6 / (k * (8 * k**2 - 3*k - 11)) * sum(list(map(lambda x: (3 * k * (k-1)-1-5*x**2) * li[x],range(k))))
    res = sum(list(map(lambda x: (c * x ** 2 + d - li[x]) ** 2, range(k))))
    return res

'''Преобразование в множество тренда'''
def delta(li, q):
    res = []
    for i in range(len(li)):
        res.append(li[i] + q*i)
    return res


'''Список минимальных расстояний'''
list_of_distances = [0]

'''Функция, находящая минимальные расстояния и проверяющая погрешность,
расстояние между кластерами рассчитано, как расстояние между центроидами(centroid linkage)'''
'''def hierarhical_centroid(df):
    flag = True
    distance_matrix = np.zeros((df.shape[0], df.shape[0]))
    for i in range(df.shape[0]):
        distance_matrix[i] = np.array(list(map(lambda x: np.linalg.norm(x - df.iloc[i, :]), np.array(df))))
    distance_df = pd.DataFrame(distance_matrix)
    distance_df.index = df.index
    distance_df.columns = df.index
    # print(distance_df)
    x_coord = 0
    y_coord = 0
    min_distance = 10 ** 5 + 1
    for i in range(distance_df.shape[0]):
        for j in range(distance_df.shape[1]):
            if distance_df.iloc[i, j] < min_distance and distance_df.iloc[i, j] != 0:
                min_distance = distance_df.iloc[i, j]
                x_coord = i
                y_coord = j
    list_of_distances.append(min_distance)
    trend = delta(list_of_distances, 0.15)
    if delta_l(trend) - delta_q(trend) > 0:
        print('характер возрастания изменился')
        flag = False
        return df, distance_df, flag
    distance_df = distance_df.drop(distance_df.columns[[x_coord, y_coord]], axis=0)
    distance_df = distance_df.drop(distance_df.columns[[x_coord, y_coord]], axis=1)
    list_of_elements = f'{df.index[x_coord]} {df.index[y_coord]}'.split()
    # print(list_of_elements)
    total = [0] * 6
    for i in list_of_elements:
        total += start_df.iloc[int(i), :]
    centroid = total / len(list_of_elements) # расстояние между кластерами представляю как расстояние между центроидами, а надо представлять как минимальное расстояние между элементами
    df.loc[f'{df.index[x_coord]} {df.index[y_coord]}'] = centroid
    df = df.drop(df.index[[x_coord, y_coord]], axis=0)
    # print(df)
    return df, distance_df, flag

result, dist, fl = hierarhical_centroid(data_2020_scale)
for _ in range(80):
    result, dist, fl = hierarhical_centroid(result)
    if fl is False:
        break
print(result)
print(dist)
print(list_of_distances)'''
# print(result.index[2].split())

# '''Построение графика последовательности минимальных расстояний'''
# x = np.arange(0, len(list_of_distances), 1)
# y = np.array(list_of_distances)
# plt.scatter(x, y)
# plt.grid(True)
# plt.show()

'''Функция, находящая минимальные расстояния и проверяющая погрешность,
расстояние между кластерами рассчитано, как минимальное расстояние между всеми элементами(single linkage)'''

def hierarhical_single(df, elements):
    flag = True
    distance_df = pd.DataFrame(np.zeros((len(elements), len(elements))))
    distance_df.index = elements
    distance_df.columns = elements
    for i in range(distance_df.shape[0]):
        obj_1 = [int(f) for f in str(distance_df.index[i]).split()]
        for j in range(distance_df.shape[0]):
            obj_2 = [int(f) for f in str(distance_df.index[j]).split()]
            # считаем расстояние между кластерами
            if obj_1 == obj_2:
                distance_df[distance_df.index[i]][distance_df.index[j]] = 0
            else:
                list_dist = []
                for k1 in obj_1:
                    for k2 in obj_2:
                        list_dist.append(np.linalg.norm(df.iloc[k1, :] - df.iloc[k2, :]))
                distance_df[distance_df.index[i]][distance_df.index[j]] = max(list_dist)
    print(distance_df)
    # ищем минимальное расстояние
    x_coord = 0
    y_coord = 0
    min_distance = 10 ** 5 + 1
    for i in range(distance_df.shape[0]):
        for j in range(distance_df.shape[1]):
            if distance_df.iloc[i, j] < min_distance and distance_df.iloc[i, j] != 0:
                min_distance = distance_df.iloc[i, j]
                x_coord = i
                y_coord = j
    list_of_distances.append(min_distance)
    # проверяем момент остановки
    trend = delta(list_of_distances, 0.2)
    if delta_l(trend) - delta_q(trend) > 0:
        print('характер возрастания изменился')
        flag = False
        return df, distance_df, flag
    new_index = f'{distance_df.index[x_coord]} {distance_df.index[y_coord]}'
    # удалим строки и столбцы в матрице сходства
    distance_df = distance_df.drop(distance_df.columns[[x_coord, y_coord]], axis=0)
    distance_df = distance_df.drop(distance_df.columns[[x_coord, y_coord]], axis=1)
    # добавим в список new_elements новый кластер
    new_elements = list(distance_df.index)
    new_elements.append(new_index)
    return distance_df, new_elements, flag

dist, elem, fl = hierarhical_single(data_2020_scale, data_2020_scale.index)
for _ in range(80):
    dist, elem, fl = hierarhical_single(data_2020_scale, elem)
    if fl is False:
        break


# '''Построение графика последовательности минимальных расстояний'''
x = np.arange(0, len(list_of_distances), 1)
y = np.array(list_of_distances)
plt.scatter(x, y)
plt.grid(True)
plt.show()

# вывод регионов в кластерах
def print_clusters(model):
  for i in np.unique(model.labels_):
    print(f'{i} cluster: {list(subjects[model.labels_ == i])}')
print_clusters(())
