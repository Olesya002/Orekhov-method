import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
for i in range(81):
    distance_df_1 = pd.DataFrame(np.zeros((df_start.shape[0], df_start.shape[0])))
    distance_df_1.index = df_start.index
    distance_df_1.columns = df_start.index
    for j in range(df_start.shape[0]):
        distance_df_1.iloc[j, :] = list(map(lambda x: np.linalg.norm(x - df_start.iloc[j, :]), np.array(df_start)))
    # print("Матрица расстояний: \n", distance_df)
    min_distance_1 = 10 ** 5 + 1
    for j in range(distance_df_1.shape[0]):
        for k in range(distance_df_1.shape[1]):
            if distance_df_1.iloc[j, k] < min_distance_1 and distance_df_1.iloc[j, k] != 0:
                min_distance_1 = distance_df_1.iloc[j, k]
                x_coord_1 = j
                y_coord_1 = k
    list_distances.append(min_distance_1)
    distance_df_1 = distance_df_1.drop(distance_df_1.columns[[x_coord_1, y_coord_1]], axis=0)
    distance_df_1 = distance_df_1.drop(distance_df_1.columns[[x_coord_1, y_coord_1]], axis=1)
    list_of_elements_1 = f'{df_start.index[x_coord_1]} {df_start.index[y_coord_1]}'.split()
    # print('Список обновлённыъ элементов: \n', list_of_elements)
    total1 = [0] * 6
    for j in list_of_elements_1:
        total1 += data_2020_scale.iloc[int(j), :]
    centroid1 = total1 / len(list_of_elements_1)
    df_start.loc[f'{df_start.index[x_coord_1]} {df_start.index[y_coord_1]}'] = centroid1
    df_start = df_start.drop(df_start.index[[x_coord_1, y_coord_1]], axis=0)
'''Список минимальных расстояний'''
list_of_distances = [0]
'''Функция, находящая минимальные расстояния и проверяющая погрешность,
расстояние между кластерами рассчитано, как минимальное расстояние между всеми элементами(single linkage)'''
def hierarhical(df):
    flag = True
    distance_df = pd.DataFrame(np.zeros((df.shape[0], df.shape[0])))
    distance_df.index = df.index
    distance_df.columns = df.index
    for i in range(df.shape[0]):
        distance_df.iloc[i, :] = list(map(lambda x: np.linalg.norm(x - df.iloc[i, :]), np.array(df)))
    # print("Матрица расстояний: \n", distance_df)
    min_distance = 10 ** 5 + 1
    for i in range(distance_df.shape[0]):
        for j in range(distance_df.shape[1]):
            if distance_df.iloc[i, j] < min_distance and distance_df.iloc[i, j] != 0:
                min_distance = distance_df.iloc[i, j]
                x_coord = i
                y_coord = j
    list_of_distances.append(min_distance)
    # print('Список минимальных расстояний: \n',list_of_distances)
    trend = delta(list_of_distances, 0.6)
    # print('Линия тренда:\n',trend)
    # проверка момента остановки процесса:
    if len(trend) >= 5:
        if delta_2(trend[len(trend) - 5:len(trend)-1]) <= 0 and delta_2(trend[len(trend) - 4:len(trend)]) > 0:
            print('характер возрастания изменился')
            print(len(list_of_distances))
            flag = False
            return df, distance_df, flag
    distance_df = distance_df.drop(distance_df.columns[[x_coord, y_coord]], axis=0)
    distance_df = distance_df.drop(distance_df.columns[[x_coord, y_coord]], axis=1)
    list_of_elements = f'{df.index[x_coord]} {df.index[y_coord]}'.split()
    # print('Список обновлённыъ элементов: \n', list_of_elements)
    total = [0] * 6
    for i in list_of_elements:
        total += data_2020_scale.iloc[int(i), :]
    centroid = total / len(list_of_elements)
    df.loc[f'{df.index[x_coord]} {df.index[y_coord]}'] = centroid
    df = df.drop(df.index[[x_coord, y_coord]], axis=0)
    # print('Матрица df \n', df)
    return df, distance_df, flag

result, dist, fl = hierarhical(data_2020_scale)
for _ in range(82):
    result, dist, fl = hierarhical(result)
    if fl is False:
        break
print(result)
print(dist)
print(len(list_of_distances))

# '''Построение графика последовательности минимальных расстояний'''
x = np.arange(0, len(list_distances), 1)
y = np.array(list_distances)
fig = plt.figure(figsize=(10, 7))
# plt.scatter(x, y, color = 'lightseagreen', s = 20)
plt.plot(x,y, linewidth = 0.8, color = 'black', marker='o', ms= 7, markerfacecolor='skyblue')
#plt.ylabel('minimal distances', fontsize=12)
#plt.axvline(x=72, color='r', linestyle='--', linewidth = 0.8)
plt.grid(True, alpha = 0.5)
plt.show()
print('Полученные кластеры:', result.index)
# вывод регионов в кластерах
def print_clusters():
  for i in np.unique(model.labels_):
    print(f'{i} cluster: {list(subjects[model.labels_ == i])}')
