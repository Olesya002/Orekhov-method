#install.packages('factoextra')
#install.packages('cluster')
#install.packages('tidyverse')
#install.packages('stringdist')
#install.packages('NbClust')
#install.packages('vegan')
install.packages('Hmisc')

# удалим строки с пустыми значени€ми, осталось 78 субъектов
#df_with_deletedNA <- na.omit(df)               


# оценка NA средним
#df$goods[is.na(df$goods)] <- mean(df$goods, na.rm = T)
#df$research[is.na(df$research)] <- mean(df$research, na.rm = T)
#df$coeff[is.na(df$coeff)] <- mean(df$coeff, na.rm = T)

#k.pam <- pam(df_scale, 4)
#fviz_silhouette(silhouette(k.pam))

# график, построенный пр€мым перебором
#wss1 <- sapply(1:8, function(k){
# kmeans(df_scale1, k, nstart = 10)$tot.withinss
#})
#plot(1:8, wss1, type = "b", pch = 19, frame = FALSE, xlab = "„исло кластеров K", ylab = "ќбща€ внутригруппова€ сумма квадратов")



# метод k-means
# определ€ем оптимальное кол-во кластеров
# использование метода "локт€"
fviz_nbclust(df_scale, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)  # определ€ем оптимальное количество кластеров
# методом локт€ дл€ 2020 года это - 4

fviz_nbclust(df_scale, kmeans, method = 'silhouette')
fviz_nbclust(df_scale, kmeans, method = 'gap_stat')

fviz_nbclust(df_scale, pam, method = "wss") 
fviz_nbclust(df_scale, pam, method = "gap_stat") 
fviz_nbclust(df_scale, pam, method = "silhouette") 



# использование статистики разрыва (k-means)
#set.seed(123)
#gap_stat1 <- clusGap(df_scale1, FUN = kmeans, nstart = 10, K.max = 8, B = 50)
#print(gap_stat1, method = "firstmax")
#fviz_gap_stat(gap_stat1)


# статистика разрыва (k-medoids)
#set.seed(123)
#gap_stat1 <- clusGap(df_scale1, FUN =  pam, K.max = 8, B = 100)
#print(gap_stat2, method = "firstmax")
#fviz_gap_stat(gap_stat1)


#k.pam <- pam(df_scale1, k = 4)

# 
install.packages('fpc')
?pamk
pamk(df_scale)  # дл€ второго он определ€ет 4
# дл€ первого метода - 6

# визуализируем
rn[c(62, 61, 82,83,85)]


#fviz_cluster(pam(df_scale, 4), stand = FALSE)


# оценки разбиени€ на кластеры
library(NbClust)
nb <- NbClust(df_scale, distance = "euclidean", min.nc = 2,
              max.nc = 8, method = "average", index = "all")
fviz_nbclust(nb) + theme_minimal()
get_clust_tendency(df_scale, n = 30,
                   gradient = list(low = "steelblue", high = "white"))


# иерархическа€ кластеризаци€
?dist
# алгомеративные методы
m1 <- dist(df_scale, method = "euclidean")   #построение матрицы дистанций 

plot(hclust(m1, method = "average"), cex = 0.7, hang = -1)  # метод средней св€зи
plot(hclust(m1, method = "single"), cex = 0.7)              # метод ближайшего соседа
plot(hclust(m1, method = "complete"), cex = 0.7)            # метод дальнего соседа






# анализ сходства дендограмм
library(dendextend)
hc1 <- hclust(m1, method = "average")
hc2 <- hclust(m1, method = "single")
hc3 <- hclust(m1, method = "complete")
hc4 <- hclust(m1, method = "ward.D2")
hc5 <- hclust(m1, method = "centroid")
dend1 <- as.dendrogram(hc1)
dend2 <- as.dendrogram(hc2)
dend3 <- as.dendrogram(hc3)
dend4 <- as.dendrogram(hc4)
dend5 <- as.dendrogram(hc5)
c(cor_cophenetic(dend1, dend2), # кофенетическа€ коррел€ци€
  cor_bakers_gamma(dend1, dend2))
c(cor_cophenetic(dend1, dend3), # кофенетическа€ коррел€ци€
  cor_bakers_gamma(dend1, dend3))
c(cor_cophenetic(dend1, dend4), # кофенетическа€ коррел€ци€
  cor_bakers_gamma(dend1, dend4))
c(cor_cophenetic(dend2, dend3), # кофенетическа€ коррел€ци€
  cor_bakers_gamma(dend2, dend3))
c(cor_cophenetic(dend2, dend4), # кофенетическа€ коррел€ци€
  cor_bakers_gamma(dend2, dend4))
c(cor_cophenetic(dend3, dend4), # кофенетическа€ коррел€ци€
  cor_bakers_gamma(dend3, dend4))


hc_list <- list(hc1, hc2, hc3, hc4, hc5)
library(vegan)
# кофенетическа€ коррел€ци€
Coph <- rbind(MantelStat <- unlist(lapply(hc_list, function(hc) mantel(m1, cophenetic(hc))$statistic)),
              MantelP <- unlist(lapply(hc_list, function(hc) mantel(m1, cophenetic(hc))$signif)))
colnames(Coph) <- c("average", "Single", "complete", "Ward.D2", "Centroid") 
rownames(Coph) <- c("W ћантел€", "–-значение")
round(Coph, 3)
# дивизионные методы
dm <- diana(df_scale1, metric = "euclidean", stand = FALSE)
plot(dm, cex = 0.7)



#fviz_cluster(clara(df_scale, 3), stand = FALSE, 
#            ellipse.type = "t", ellipse.level = 0.6)

library('fpc')
Dbscan_cl <- dbscan(df_scale, eps = 0.45, MinPts = 5) 
plot(Dbscan_cl, df_scale, main = "DBScan") 