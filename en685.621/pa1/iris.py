# -----------------------------------------------------------
# iris.py
# 
# Jake Sciotto
# EN685.621 Algorithms for Data Science
# Johns Hopkins University
# Summer 2020
# -----------------------------------------------------------

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import util

# -----------------------------------------------------------
# Preparation
# -----------------------------------------------------------
iris = load_iris()
dataset = pd.DataFrame(iris['data'], columns=iris['feature_names'])
dataset.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
dataset['target'] = iris['target']
dataset['class']  = dataset['target'].apply(lambda x: iris['target_names'][x])

f = open("output/output.txt", "w")

# -----------------------------------------------------------
# Visualization
# -----------------------------------------------------------
plt.figure(0)

scatter = plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')
plt.legend(*scatter.legend_elements(), loc="upper right", title="Class")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# -----------------------------------------------------------
# Sorting
# -----------------------------------------------------------
for i in range(0, 4):
	s = "Sorting by: " + str(dataset.columns[i]) + "\n" + util.LINE + "\n"
	f.write(s)
	util.insertion_sort(dataset.values, i, f)

# -----------------------------------------------------------
# Outliers
# -----------------------------------------------------------
f.write("\nBefore removing outliers \n")
util.get_data_stats(dataset, f)

# set up plots
fig, axs = plt.subplots(4, 3)
plt.figure(1)

# sepal length vs all
axs[0, 0].scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')
axs[0, 1].scatter(dataset.iloc[:, 0], dataset.iloc[:, 2], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')
axs[0, 2].scatter(dataset.iloc[:, 0], dataset.iloc[:, 3], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')

# sepal width vs all
axs[1, 0].scatter(dataset.iloc[:, 1], dataset.iloc[:, 0], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')
axs[1, 1].scatter(dataset.iloc[:, 1], dataset.iloc[:, 2], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')
axs[1, 2].scatter(dataset.iloc[:, 1], dataset.iloc[:, 3], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')

# petal length vs all
axs[2, 0].scatter(dataset.iloc[:, 2], dataset.iloc[:, 0], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')
axs[2, 1].scatter(dataset.iloc[:, 2], dataset.iloc[:, 1], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')
axs[2, 2].scatter(dataset.iloc[:, 2], dataset.iloc[:, 3], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')

# petal width vs all
axs[3, 0].scatter(dataset.iloc[:, 3], dataset.iloc[:, 0], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')
axs[3, 1].scatter(dataset.iloc[:, 3], dataset.iloc[:, 1], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')
axs[3, 2].scatter(dataset.iloc[:, 3], dataset.iloc[:, 2], c=dataset['target'], cmap=plt.cm.Set1, edgecolor='k')

# remove outliers 
new_dataset = util.remove_outliers(dataset)

# look at skew and mean of new dataset without outliers
f.write("\nAfter removing outliers\n")
util.get_data_stats(new_dataset, f)

plt.figure(2)
scatter = plt.scatter(new_dataset.iloc[:, 0], new_dataset.iloc[:, 1], c=new_dataset['target'], cmap=plt.cm.Set1, edgecolor='k')
plt.legend(*scatter.legend_elements(), loc="upper right", title="Class")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()
# -----------------------------------------------------------
# Feature Ranking
# -----------------------------------------------------------
array = new_dataset.values
names = new_dataset.columns[0:4]
features = array.shape[1] - 2

# separate data by features and class label
X = array[:, 0:4]
Y = array[:, 5]

# pick two features to compare
feature_selection = ['setosa', 'versicolor']

# empty array of zeros to hold distances
bh_cont_dist = [0] * features
bh_theo_dist = [0] * features

# find bhattacharyya distances of features and add result to bh_distances array
for i, name in enumerate(names):
	X1 = np.array(X[:, i], dtype = np.float64)[Y == feature_selection[0]]
	X2 = np.array(X[:, i], dtype = np.float64)[Y == feature_selection[1]]
	bh_cont_dist[i] = util.bhatta_cont(X1, X2)
	bh_theo_dist[i] = util.bhatta_theo(X1, X2)

# print corresponding distances for each feature
f.write("\nMethod: continuous density function\n")
f.write("Feature ranking:\n")
for n, d in sorted(zip(names, bh_cont_dist)):
	distance = str("Bhattacharyya distance for: ") + str(n) + " " + str(d) + "\n"
	f.write(distance)

f.write("\nMethod: theoretical calculation\n")
f.write("Feature ranking:\n")
for n, d in sorted(zip(names, bh_theo_dist)):
	distance = str("Bhattacharyya distance for: ") + str(n) + " " + str(d) + "\n"
	f.write(distance)