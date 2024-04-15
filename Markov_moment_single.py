import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


'''Вычисление погрешностей аппроксимации, для 4-х точек'''
def delta_2(li):
    # минимальные расстояния, перенесённые в начало координат
    new_li = np.array(li) - li[0]
    res = (1/245) * (19 * new_li[1] ** 2 - 11 * new_li[2] ** 2 + 41 * new_li[3] ** 2 + 12*new_li[1]*new_li[2]-64*new_li[1]*new_li[3]-46*new_li[2]*new_li[3])
    return res

'''Преобразование в множество тренда'''
def delta(li, q):
    res = []
    for i in range(len(li)):
        res.append(li[i] + q*i)
    return res


list_distances = [0]
df_start = data_2020_scale.copy()
elements = df_start.index
for i in range(81):
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
                        list_dist.append(np.linalg.norm(df_start.iloc[k1, :] - df_start.iloc[k2, :]))
                distance_df[distance_df.index[i]][distance_df.index[j]] = max(list_dist)
    x_coord = 0
    y_coord = 0
    min_distance = 10 ** 5 + 1
    for i in range(distance_df.shape[0]):
        for j in range(distance_df.shape[1]):
            if distance_df.iloc[i, j] < min_distance and distance_df.iloc[i, j] != 0:
                min_distance = distance_df.iloc[i, j]
                x_coord = i
                y_coord = j
    list_distances.append(min_distance)
    trend = delta(list_distances, 0.3)
    if len(trend) >= 5:
        if delta_2(trend[len(trend) - 5:len(trend) - 1]) <= 0 and delta_2(trend[len(trend) - 4:len(trend)]) > 0:
            print('характер возрастания изменился')
            print(len(list_distances))
            print(distance_df)
    new_index = f'{distance_df.index[x_coord]} {distance_df.index[y_coord]}'
    # удалим строки и столбцы в матрице сходства
    distance_df = distance_df.drop(distance_df.columns[[x_coord, y_coord]], axis=0)
    distance_df = distance_df.drop(distance_df.columns[[x_coord, y_coord]], axis=1)
    # добавим в список new_elements новый кластер
    elements = list(distance_df.index)
    elements.append(new_index)


# '''Построение графика последовательности минимальных расстояний'''
x = np.arange(0, len(list_distances), 1)
y = np.array(list_distances)
# plt.scatter(x, y, color = 'lightseagreen', s = 20)
fig = plt.figure(figsize=(10, 7))
plt.plot(x,y, linewidth = 0.8, color = 'black', marker='o', ms= 7, markerfacecolor='skyblue')
plt.axvline(x=75, color='r', linestyle='--', linewidth = 1)
#plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
#plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
#plt.ylabel('minimal distances', fontsize=12)
plt.grid(True, alpha = 0.5)
plt.show()

