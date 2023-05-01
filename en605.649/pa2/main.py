# -----------------------------------------------------------
# main.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import knn, util, cnn, enn

# import classification datasets
glass = pd.read_csv("input/glass.data")
segs  = pd.read_csv("input/segmentation.data")
votes = pd.read_csv("input/house-votes-84.data")

# import regression datasets
abalone = pd.read_csv("input/abalone.data")
machine = pd.read_csv("input/machine.data")
fires   = pd.read_csv("input/forestfires.data")


###################################
# Data pre-processing
###################################

# Glass dataset
#
# Changes: Columns renamed, id column dropped
glass.columns = ["id", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "y"]
glass.drop(["id"], axis = 1, inplace = True)

# Segmentation dataset
#
# changes: Class column encoded, index reset
segs["y"] = segs.index
segs["y"] = segs["y"].astype("category")
segs["y"] = segs["y"].cat.codes
segs = segs.reset_index(drop = True)

# Votes dataset
#
# Changes: column names, missing values, class column encoded and moved
votes.columns = ["class", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
				 "x11", "x12", "x13", "x14", "x15", "x16"]

votes.replace("?", 2, inplace = True)
votes.replace("y", 1, inplace = True)
votes.replace("n", 0, inplace = True)

# move republican/democrat column to the end 
votes["y"] = votes["class"].apply(lambda x: 1 if x == "republican" else 0)
votes.drop(["class"], axis = 1, inplace = True)

print(votes.head())

# REGRESSION DATASETS
# ---------------------------------

# Abalone dataset
#
# Changes: First column to categorical from nominal, in alphabetic order
abalone.columns = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "y"]
abalone["x1"] = abalone["x1"].apply(lambda x: 1 if x == "F" else 2 if x == "I" else 3)

# Machine dataset
#
# Changes: ERP feature, first two columns need to be dropped
machine.drop(machine.columns[[0, 1, 9]], axis = 1, inplace = True)
machine.columns = ["x1", "x2", "x3", "x4", "x5", "x6", "y"]

# Forest Fires dataset
# Changes: Log transform on Area feature
# 
# Notes: Several features are correlated, needs 1) feature selection and log transform
# 	     and month, day columns were encoded

fires["month"] = fires["month"].astype("category")
fires["month"] = fires["month"].cat.codes
fires["day"] = fires["day"].astype("category")
fires["day"] = fires["day"].cat.codes

print(abalone)
print(machine)
print(fires)


figure = plt.figure()

axes = figure.add_subplot(1, 2, 1)
axes.set_title("Area Before Log Transform")
axes.hist(fires['area'], color = "dimgray")

# perform log transform and add 1 to prevent -inf error
fires['area'] = np.log(fires['area'] + 1)

axes = figure.add_subplot(1, 2, 2)
axes.set_title("Area After Log Transform")
axes.hist(fires['area'], color = "dimgray")

plt.show()

X = [0, .1, .1, 0, 0, 0, .5, 0, 0, 0]
y = [0, 0, 0, .2, 0, 0, .2, .1, 0, 0]

print("Demonstration of Euclidean distance: ", knn.euclidean_distance(X, y))
print("Demonstration of Gaussian kernel: ", knn.gaussian(knn.euclidean_distance(X, y), .05))

#########################################################
# K-Nearest Neighbor Classification (Normal)
#########################################################

names = ["Glass", "Segmentation", "Votes"]
frames = [glass, segs, votes]

n_folds = 5
nb_neighbors = 3

print("\n---------- KNN: Normal ----------")

for name, frame in zip(names, frames):

	start = datetime.now()

	# with min-max normalization
	new_frame = knn.normalize(frame.values)
	x_train, y_train, x_test, y_test = util.shuffle_split_data(new_frame.values, new_frame.values)

	cv_scores = knn.evaluate(x_train, n_folds, nb_neighbors, False)

	print("Dataset: ", name)
	print("Scores: ", cv_scores)
	print("Mean accuracy: \t", sum(cv_scores) / float(len(cv_scores)))
	print("Running time: \t", datetime.now() - start)
	print("")

#########################################################
# K-Nearest Neighbor Classification (Condensed)
#########################################################

# removing correct examples, because they might be unnecessary for classification

print("---------- KNN: Condensed ----------")

names = ["Glass", "Segmentation", "Votes"]
frames = [glass, segs, votes]

for name, frame in zip(names, frames):

	start = datetime.now()

	# with minmax normalization
	new_frame = knn.normalize(frame.values)

	condensed_x, condensed_y = [], []

	X = frame[frame.columns[:-1]]
	y = frame[frame.columns[-1]]

	x_train, y_train, x_test, y_test = util.shuffle_split_data(X.values, y.values)

	condensed = cnn.condense(x_train, y_train)
	for i in condensed:
		condensed_x.append(x_train[i])
		condensed_y.append(y_train[i])

	condensed_train_temp = np.array(condensed_x)

	predicted_y = cnn.test_knn(condensed_train_temp, condensed_y, x_test, nb_neighbors)
	accuracy = cnn.get_accuracy(y_test, predicted_y)

	print("Dataset: \t", name)
	print("Accuracy: \t", accuracy)
	print("Running time: \t", datetime.now() - start)
	print("")

#########################################################
# K-Nearest Neighbor Classification (Edited)
#########################################################

print("---------- KNN: Edited ----------")

names = ["Glass", "Segmentation", "Votes"]
frames = [glass, segs, votes]


for name, frame in zip(names, frames):

	start = datetime.now()

	X = frame[frame.columns[:-1]]
	y = frame[frame.columns[-1]]

	print("Original dataset length: \t", len(X))

	edited = enn.edit(frame, y, nb_neighbors)
	print("New dataset length: \t", len(edited))

	# calculate the percentage reduction
	reduction = float(len(edited) / len(X))

	new_frame = edited.drop(["correct", "predicted", "result"], axis = 1, inplace = True)

	cv_scores = knn.evaluate(edited.values, n_folds, nb_neighbors, False)
	print("Dataset: \t", name)
	print("Percent reduction: \t", round(100 - (reduction * 100), 3))
	print("CV Scores: \t", cv_scores)
	print("Mean accuracy: \t", (sum(cv_scores) / float(len(cv_scores))))
	print("Running time: \t", datetime.now() - start)
	print("")

#########################################################
# K-Nearest Neighbor Regression 
#########################################################

names = ["Abalone", "Machine", "Fires"]
frames = [abalone, machine, fires]

print("---------- KNN: Regression ----------")

for name, frame in zip(names, frames):

	start = datetime.now()

	X = frame.iloc[:, -1:]
	y = frame.iloc[:, :-1]

	x_train, y_train, x_test, y_test = util.shuffle_split_data(X.values, y.values)
	cv_scores = knn.evaluate(x_train, n_folds, nb_neighbors, True)

	print("Dataset: \t", name)
	print("CV Scores: \t", cv_scores)
	print("MSE: \t", (sum(cv_scores) / float(len(cv_scores))))
	print("Running time: \t", datetime.now() - start)
	print("")

