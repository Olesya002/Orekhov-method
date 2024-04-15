library(cluster)
library(factoextra)
library(readxl)
library(tidyverse)
library(magrittr)
library(dplyr)
library(broom)
library(stringdist)
library(Hmisc)
library(fpc)
?read_excel

# считываем данные 2020 года
df_2020 <- read_xlsx('main_data.xlsx', sheet = '2020')     
df_2020 <- df_2020[,0:7]
colnames(df_2020) <- c("subject", "VRP", "investments", "funds", "coeff", "research", "goods")

# считываем данные 2019 года
df_2019 <- read_xlsx('main_data.xlsx', sheet = '2019')
df_2019 <- df_2019[,0:7]
colnames(df_2019) <- c("subject", "VRP", "investments", "funds", "coeff", "research", "goods")
df_2019 <- df_2019[-c(22, 23, 63, 64, 65),]


# функци€ дл€ считывани€ данных 2018 - 2014 гг
read_data <- function(s){
  df <- read_xlsx('main_data.xlsx', sheet = s)
  df <- df[-c(1,20,24,25,33,42,50,65,69,70,71,73,84),0:7]
  colnames(df) <- c("subject", "VRP", "investments", "funds", "coeff", "research", "goods")
  df$coeff <- as.double(df$coeff)
  df <- subset(df, select = -c(subject))
  return(df)
}

df_2018 <- read_data('2018')
df_2017 <- read_data('2017')
df_2016 <- read_data('2016')
df_2015 <- read_data('2015')
df_2014 <- read_data('2014')

# функци€ дл€ 2013 - 2011 гг
read_data_1 <- function(s){
  df <- read_xlsx('main_data.xlsx', sheet = s)
  df <- df[-c(1,20,24,25,33,36, 41,42,50,65,69,70,71,73,84),0:7]
  colnames(df) <- c("subject", "VRP", "investments", "funds", "coeff", "research", "goods")
  df$coeff <- as.double(df$coeff)
  return(df)
}
df_2013 <- read_data_1('2013')
df_2012 <- read_data_1('2012')
df_2011 <- read_data_1('2011')

# список субъектов
subjects <- c(df_2020$subject)  #названи€ субъектов 82 шт.
subj_without <- c(df_2013$subject) #названи€ субъектов 80 шт. без  рыма

# удаление перового столбца
df_2020 <- subset(df_2020, select = -c(subject))
df_2019 <- subset(df_2019, select = -c(subject))
df_2013 <- subset(df_2013, select = -c(subject))
df_2012 <- subset(df_2012, select = -c(subject))
df_2011 <- subset(df_2011, select = -c(subject))

# стандартизируем все наблюдени€ 
df_scale_2020 <- scale(df_2020)        
df_scale_2019 <- scale(df_2019) 
df_scale_2018 <- scale(df_2018)
df_scale_2017 <- scale(df_2017)
df_scale_2016 <- scale(df_2016)
df_scale_2015 <- scale(df_2015)
df_scale_2014 <- scale(df_2014)
df_scale_2013 <- scale(df_2013)
df_scale_2012 <- scale(df_2012)
df_scale_2011 <- scale(df_2011)


#kmeans:
# применение метода n раз
use_method <- function(n, df){
  #список моделей
  list_of_models <- list()
  for (i in 1:n) {
    km <- kmeans(df, centers = 4)
    list_of_models[[i]] = km
  }
  return(list_of_models)
}

#какое разбиение встречаетс€ чаще всего
find_frequent <- function(models){
  #список с размерами кластеров дл€ каждой модели
  list_of_sizes <- lapply(models, function(a) a['size'])
  #список с отсортированными размерами кластеров
  sorted_list <- lapply(list_of_sizes, function(s) {
    l <- sort(unlist(s))
    names(l) <- NULL
    return(l)
  })
  #список уникальных разбиений
  unique_list <- unique(sorted_list)
  #подсчитаем сколько раз было осуществлено такое разбиение
  razb <- list()
  for (i in 1:length(unique_list)) {
    razb[[i]] = sum(unlist(lapply(sorted_list, function(s) all(s == unique_list[[i]]))))
  }
  return(cbind(unique_list, razb))
}


# 2020 year
models_2020 <- use_method(20, df_scale_2020)
result <- find_frequent(models_2020)

# определ€ю модель, дающую такое разбиение
lapply(models_2020, function(s) s['size'])
# визуализирую еЄ
fviz_cluster(models_2020[[2]], data = df_scale_2020)
# между прочим такое разбиение выгл€дит лучше всех на графике


# 2019 year
models_2019 <- use_method(20, df_scale_2019)
result <- find_frequent(models_2019)
result[2,1]
# определ€ю модель, дающую такое разбиение
lapply(models_2019, function(s) s['size'])
# визуализирую еЄ
fviz_cluster(models_2019[[3]], data = df_scale_2019)








# kmeans 2019
km_2019 <- kmeans(df_scale_2019, centers = 4)
fviz_cluster(km_2019, data = df_scale_2019)

subjects[unlist(lapply(km_2019['cluster'], function(s) which(s == 4)))]



?kmeans
# тут удобное определение регионов, вход€щих в кластер
first <-  lapply(models_2019[[3]]['cluster'], function(s) which(s == 1))
second <- lapply(models_2019[[3]]['cluster'], function(s) which(s == 2))
third <- lapply(mvp['cluster'], function(s) which(s == 3))
fourth <- lapply(mvp['cluster'], function(s) which(s == 4))
subjects[unlist(second)]





