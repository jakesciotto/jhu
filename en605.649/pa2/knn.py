# -----------------------------------------------------------
# knn.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------
import numpy as np
import pandas as pd
import random
import math
import util

"""
Three steps to KNN:
	1. Get Euclidean distances
	2. Get nearest neighbors
	3. Make predictions
"""

def get_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

def gaussian(dist, sigma):
	"""
	Gaussian weighted distance for regression.
	"""
	return 1./(math.sqrt(2. * math.pi) * sigma) * math.exp(-dist ** 2/ (2 * sigma ** 2))

def euclidean_distance(row1, row2):
	"""
	Calculation of Euclidean distance for KNN.
	"""

	return np.linalg.norm(np.array(row1) - np.array(row2))
	"""
	distance = 0.0
	for i in range(len(row1) - 1):
		distance += (row1[i] - row2[i]) ** 2
	return math.sqrt(distance)
	"""

def get_neighbors(training, testing_row, nb_neighbors):
	"""
	Get nearest neighbors for KNN classiication.
	"""
	distances = list()
	for training_row in training:
		d = euclidean_distance(testing_row, training_row)
		distances.append((training_row, d))
	distances.sort(key = lambda x: x[1])
	neighbors = list()
	for i in range(nb_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def get_reg_neighbors(training, testing_row, nb_neighbors):
	"""
	Get nearest neighbors for KNN regression.
	"""
	distances = list()
	for training_row in training:
		d = euclidean_distance(testing_row, training_row)
		new_d = gaussian(d, .01)
		distances.append((training_row, new_d))
	distances.sort(key = lambda x: x[1])
	neighbors = list()
	for i in range(nb_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def predict(training, testing_row, nb_neighbors):
	"""
	Make a classification prediction.
	"""
	neighbors = get_neighbors(training, testing_row, nb_neighbors)
	result = [row[-1] for row in neighbors]
	return max(set(result), key = result.count)

def predict_reg(training, testing_row, nb_neighbors):
	"""
	Make a regression prediction.
	"""
	neighbors = get_reg_neighbors(training, testing_row, nb_neighbors)
	result = [row[-1] for row in neighbors]
	return sum(result) / float(len(result))

def normalize(dataset):
	"""
	Performs minmax normalization on values.
	"""
	dataset_normalized = (dataset - dataset.min()) / (dataset.max() - dataset.min())
	return pd.DataFrame(data = dataset_normalized)

def cross_val(dataset, folds):
	""" 
	Perform cross-validation split of dataset before KNN.
	"""
	split_data = list()
	copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		# shuffles the dataset
		while len(fold) < fold_size:
			index = random.randrange(len(copy))
			fold.append(copy.pop(index))
		split_data.append(fold)
	print("Fold size: ", fold_size)
	return split_data

def evaluate(dataset, n_folds, nb_neighbors, flag):
	"""
	Evaluate the algorithm with cross validation scores.
	"""
	folds = cross_val(dataset, n_folds)

	scores = list()
	for fold in folds:
		training = list(folds)
		# I believe I'm getting a Python2 ValueError for remove() 
		try:
			training.remove(fold)
		except ValueError:
			pass
		training = sum(training, [])
		testing = list()

		for row in fold:
			copy = list(row)
			testing.append(copy)
			# this was None but Euclidean distance was not calculating
			copy[-1] = 0.0

		preds = knn(training, testing, nb_neighbors, flag)
		actual = [row[-1] for row in fold]
		# flag for regression
		if flag:
			# THIS IS WHERE TUNING GOES
			accuracy = mean_squared_error(actual, preds)
		else:
			accuracy = get_cv_accuracy(actual, preds)
		scores.append(accuracy)
	return scores

def knn(training, testing, nb_neighbors, flag):
	"""
	Calls the KNN algorithm.
	"""
	preds = list()
	for row in testing:
		if flag:
			output = predict_reg(training, row, nb_neighbors)
		else:
			output = predict(training, row, nb_neighbors)
		preds.append(output)
	print("Neighbors: ", row)
	print("Output: ", output)
	return preds

def get_cv_accuracy(test, predictions):
	"""
	Determine accuracy rate for cross val scores.
	"""
	correct = 0
	for i in range(len(test)): 
		if test[i] == predictions[i]: 
			correct += 1
	return (correct / float(len(test))) * 100.0

def get_accuracy(test, predictions): 
	""" 
	Determine accuracy rate.
	"""
	correct = 0
	for i in range(len(test)): 
		if test[i][-1] == predictions[i]: 
			correct += 1
	return (correct / float(len(test))) * 100.0

def mean_squared_error(actual, predicted):
	"""
	Returns mean squared error for regression datasets.
	"""
	sum_error = 0.0
	for i in range(len(actual)):
		difference = predicted[i] - actual[i]
		sum_error += (difference ** 2)
	mean_error = sum_error / float(len(actual))
	return mean_error

