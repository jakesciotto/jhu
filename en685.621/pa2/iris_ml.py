# -----------------------------------------------------------
# iris_ml.py
# 
# Jake Sciotto
# EN685.621 Algorithms for Data Science
# Johns Hopkins University
# Summer 2020
# -----------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import util

# -------------------------------------------------------------
# Data Cleansing
# -------------------------------------------------------------
dataset = pd.read_csv("input/iris_6_features_for_cleansing.csv")
dataset.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'feature-5', 'feature-6', 'class']
target_names = ['setosa', 'versicolor', 'virginica']

# check for null values
blanks = util.find_blanks(dataset)

rows, cols = dataset.shape[0], dataset.shape[1]

# filling the blanks with the median for the time being
for i in range(0, cols):
	dataset.iloc[:, i].fillna((dataset.iloc[:, i].median()), inplace=True)

# initial visualization
plt.figure(0)
plt.title("Initial visualization")
scatter = plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c = dataset['class'], cmap = plt.cm.Set1, edgecolor = 'k')
plt.legend(*scatter.legend_elements(), loc="best", title="Class")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

f = open("output/output.txt", "w")

# -------------------------------------------------------------
# Feature Generation
# -------------------------------------------------------------

# picking top two features [petal-length], [petal-width] (from pa1) to generate from
X = dataset.iloc[:, 2:4]

# separate classes 
setosa = X[dataset['class'] == 1]
versi = X[dataset['class'] == 2]
virgi = X[dataset['class'] == 3]

# new lists for stats data
setosa_mean, versi_mean, virgi_mean, setosa_std, versi_std, virgi_std, setosa_cov, versi_cov, virgi_cov = ([] for i in range(9))

# mean 
for i in range(0, 2):
	setosa_mean.append(round(np.mean(setosa.iloc[:, i]), 4))
	versi_mean.append(round(np.mean(versi.iloc[:, i]), 4))
	virgi_mean.append(round(np.mean(virgi.iloc[:, i]), 4))

# standard deviation
for i in range(0, 2):
	setosa_std.append(round(np.std(setosa.iloc[:, i], ddof=1), 4))
	versi_std.append(round(np.std(versi.iloc[:, i], ddof=1), 4))
	virgi_std.append(round(np.std(virgi.iloc[:, i], ddof=1), 4))

# covariance
setosa_cov = setosa.iloc[:, 0:2].cov()
versi_cov = versi.iloc[:, 0:2].cov()
virgi_cov = virgi.iloc[:, 0:2].cov()

# generate additional observations
new_setosa = pd.DataFrame(np.random.random_sample((50, 2)))
new_versi = pd.DataFrame(np.random.random_sample((50, 2)))
new_virgi = pd.DataFrame(np.random.random_sample((50, 2)))

# generate new lists for std and mean of new values
new_setosa_mean, new_setosa_std, new_versi_mean, new_versi_std, new_virgi_mean, new_virgi_std = ([] for i in range (6))

# copy
setosa_scaled = new_setosa.copy()
versi_scaled = new_versi.copy()
virgi_scaled = new_virgi.copy()

# find new means
for i in range(0, 2):
	new_setosa_mean.append(np.mean(new_setosa.iloc[:, i]))
	new_versi_mean.append(np.mean(new_versi.iloc[:, i]))
	new_virgi_mean.append(np.mean(new_virgi.iloc[:, i]))

# find new stds
for i in range(0, 2):
	new_setosa_std.append(np.std(new_setosa.iloc[:, i], ddof=1))
	new_versi_std.append(np.std(new_versi.iloc[:, i], ddof=1))
	new_virgi_std.append(np.std(new_virgi.iloc[:, i], ddof=1))

# z-score normalization
for i in range(0, 50):
	for j in range(0, 2):
		setosa_scaled.iloc[i, j] = (setosa_scaled.iloc[i, j] - new_setosa_mean[j]) / new_setosa_std[j]
		versi_scaled.iloc[i, j] = (versi_scaled.iloc[i, j] - new_versi_mean[j]) / new_versi_std[j]
		virgi_scaled.iloc[i, j] = (virgi_scaled.iloc[i, j] - new_virgi_mean[j]) / new_virgi_std[j]

# multiply by the covariance
setosa_scaled = setosa_scaled.dot(setosa_cov.values)
versi_scaled = versi_scaled.dot(versi_cov.values)
virgi_scaled = virgi_scaled.dot(virgi_cov.values)

# add back the mean of the original data to scale data correctly
for i in range(0, 50):
	for j in range(0, 2):
		setosa_scaled.iloc[i, j] = setosa_scaled.iloc[i, j] + setosa_mean[j]
		versi_scaled.iloc[i, j] = versi_scaled.iloc[i, j] + versi_mean[j]
		virgi_scaled.iloc[i, j] = virgi_scaled.iloc[i, j] + virgi_mean[j]

# look at the generated features 
frames = [setosa_scaled, versi_scaled, virgi_scaled]
result = pd.concat(frames).reset_index(drop = True)

plt.figure(1)
plt.title("Newly generated feature")
scatter1 = plt.scatter(versi['petal-length'], versi['petal-width'], c='r')
scatter2 = plt.scatter(versi_scaled.iloc[:, 0], versi_scaled.iloc[:, 1], c='b')
plt.legend(['Old feature', 'New feature'], loc="best", title="Class")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# insert new features into dataframe
dataset.insert(6, 'feature-7', result[0])
dataset.insert(7, 'feature-8', result[1])

# -------------------------------------------------------------
# Feature Preprocessing / Outlier Removal
# -------------------------------------------------------------

# visualization with box plot to see outliers
plt.figure(2)
plt.title("Boxplot showing outliers")
dataset.boxplot()
plt.xticks(rotation = 45)

Q1 = dataset.quantile(.25)
Q3 = dataset.quantile(.75)
IQR = Q3 - Q1

"""
This section is commeted out, but it was to explore changing out the outliers
and seeing where they were located. I found that there were really only 
visible outliers in the sepal width class and I decided to move forward without
removing them. I think they're a principal part of the dataset. There is even 
some speculation that they were generated by a different process before being
added to the dataset.

#print(dataset < (Q1 - 1.5 * IQR)) or (dataset > (Q3 + 1.5 * IQR))

median = dataset['sepal-width'].median()

# replace the outliers with the median
#dataset['sepal-width'] = np.where(dataset['sepal-width'] <= dataset['sepal-width'].quantile(.05), median, dataset['sepal-width'])
#dataset['sepal-width'] = np.where(dataset['sepal-width'] >= dataset['sepal-width'].quantile(.95), median, dataset['sepal-width'])

#print(dataset < (Q1 - 1.5 * IQR)) or (dataset > (Q3 + 1.5 * IQR))
#plt.subplot(1, 2, 2) 
#plt.title("After removing outliers")
#dataset.boxplot()
#plt.xticks(rotation = 45)
"""

# -------------------------------------------------------------
# Feature Ranking
# -------------------------------------------------------------
array = dataset.values
names = dataset.columns[0:8]
features = array.shape[1] - 1

# separate data by features and class label
X = array[:, 0:8]
Y = array[:, 8]

# comparing first two classes
feature_selection = [1, 2]

# distances
bh_dist = [0] * features

# find bhattacharyya distances of features and add result to bh_dist array
for i, name in enumerate(names):
	X1 = np.array(X[:, i], dtype = np.float64)[Y == feature_selection[0]]
	X2 = np.array(X[:, i], dtype = np.float64)[Y == feature_selection[1]]
	bh_dist[i] = util.bhatta_cont(X1, X2)

# show distances
f.write("Feature ranking\n")
f.write(util.LINE)
for n, d in sorted(zip(names, bh_dist), key = lambda x: x[1], reverse = True):
	distance = str("Bhattacharyya distance for: ") + str(n) + " " + str(d) + "\n"
	f.write(distance)

# -------------------------------------------------------------
# Principal Component Analysis
# -------------------------------------------------------------
x = dataset.loc[:, names].values
# data has to be scaled 
x = StandardScaler().fit_transform(x)

pca = PCA(n_components = 2)

principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data = principal_components, columns = ['principal-component-1', 'principal-component-2'])
final_df = pd.concat([principal_df, dataset[['class']]], axis = 1)

plt.figure(4)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 component PCA')

targets = [1, 2, 3]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = final_df['class'] == target
    plt.scatter(final_df.loc[indicesToKeep, 'principal-component-1'], final_df.loc[indicesToKeep, 'principal-component-2'], c = color, s = 50)
plt.legend(targets, loc = 'best')

# -------------------------------------------------------------
# Machine Learning Techniques
# -------------------------------------------------------------

f.write("Machine learning techniques:\n")
f.write(util.LINE)

################################################
# Expectation maximization
################################################
gmm = GaussianMixture(n_components = 3)

X = dataset.iloc[:, :2]
gmm.fit(X)

labels = gmm.predict(X)

# split up new dataframe by labels
X['labels'] = labels
d0 = X[X['labels'] == 0]
d1 = X[X['labels'] == 1]
d2 = X[X['labels'] == 2]

preds = pd.concat([d0, d1, d2]).reset_index()
preds['labels'] = preds['labels'] + 1

# count how many correct predictions we have 
correct_preds = np.where(dataset['class'] == preds['labels'], True, False)
accuracy = np.count_nonzero(correct_preds) / 150

# plot
plt.figure(5)
plt.title("Expectation maximization")
plt.scatter(d0.iloc[:, 0], d0.iloc[:, 1], edgecolors ='r', facecolors = "none", marker = "o") 
plt.scatter(d1.iloc[:, 0], d1.iloc[:, 1], edgecolors ='b', facecolors = "none", marker = "o") 
plt.scatter(d2.iloc[:, 0], d2.iloc[:, 1], edgecolors ='g', facecolors = "none", marker = "o")
plt.scatter(X.iloc[0:50, 0], X.iloc[0:50, 1], c = 'r', marker = "x")
plt.scatter(X.iloc[50:100, 0], X.iloc[50:100, 1], c = 'b', marker = "x")
plt.scatter(X.iloc[100:150, 0], X.iloc[100:150, 1], c = 'g', marker = "x")

f.write("\nMeans for EM:\n") 
f.write(str(gmm.means_))
f.write("\nLower bound for EM:\n") 
f.write(str(gmm.lower_bound_))
f.write("\nIterations to convergence:\n") 
f.write(str(gmm.n_iter_) + "\n")
f.write("\nNumber of correct predictions\n")
f.write(str(accuracy))

################################################
# Linear Discriminant Analysis
################################################
X = dataset.iloc[:, 0:7]
y = dataset.iloc[:, 8]

lda = LinearDiscriminantAnalysis(n_components = 2)
X_r = lda.fit(X, y).transform(X)

# plot
colors = ['navy', 'turquoise', 'darkorange']
plt.figure(6)
plt.title('LDA of IRIS dataset')
for color, i, target_name in zip(colors, [1, 2, 3], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha = .8, color = color, label = target_name)
plt.legend(loc = 'best', shadow = False, scatterpoints = 1)


################################################
# MLPClassifier
################################################
# training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# try the perceptron first
per = Perceptron(random_state = 1, max_iter = 30, tol = 0.001)
per.fit(X_train, y_train)

yhat_train_per = per.predict(X_train)
yhat_test_per = per.predict(X_test)

f.write("\nPerceptron prediction\n")
f.write(util.LINE)
f.write(str(accuracy_score(y_train, yhat_train_per)) + "\n")
f.write(str(accuracy_score(y_test, yhat_test_per)) + "\n")

"""
Parameter choices:

- 50 iterations 
- Stochastic gradient descent solver with a .1 learning rate, SGD converges well
and we do not want to end up on the other side of the function
- Activation function tanh converges well even on large datasets

Hidden layers not used but can be specifed by:

N_h = N_s / (alpha * (N_i + N_o)

N_s = amount of samples in training data
N_i = input layer neurons (features)
N_o = output layer neurons 
alpha = scaling constant
"""

mlp = MLPClassifier(max_iter = 50, alpha = 1e-5, solver = 'sgd', verbose = 10, random_state = 1,
					learning_rate_init = .1, activation = 'tanh')
mlp.fit(X_train, y_train)

yhat_train_mlp = mlp.predict(X_train)
yhat_test_mlp = mlp.predict(X_test)

f.write("\nMLPClassifier\n")
f.write(util.LINE)
f.write(str(accuracy_score(y_train, yhat_train_mlp)) + "\n")
f.write(str(accuracy_score(y_test, yhat_test_mlp)) + "\n")

################################################
# SVM 
################################################
X = dataset.iloc[:, 0:2]
y = dataset.iloc[:, 8]
h = 0.02

# linear kernel 
model = svm.SVC(kernel = 'linear', C = 1.0).fit(X, y)

x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# color plot
plt.figure(7)
plt.contour(xx, yy, Z)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c = y, edgecolors = 'k')
plt.show()