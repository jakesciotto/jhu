# -----------------------------------------------------------
# cnn.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

import numpy as np
import random as r
import operator
from collections import Counter

def test_knn(training_x, training_y, testing_x, k):
	"""
	Runs KNN on the data.
	"""
	predictions = [] 
	for y in range(len(testing_x)):
		distance = []

		# euclidean distance calculation
		distance = np.linalg.norm(training_x - testing_x[y], axis = 1)

		temp = zip(distance, training_y) 
		sorted_temp = sorted(temp, key = lambda x: x[0])
		nn = sorted_temp[:k]
		nn_class = map(operator.itemgetter(1), nn)
		c = Counter(nn_class).most_common()
		predictions.append(c[0][0])
	return predictions

def condense(training_x, training_y):
	"""
	Condenses the training data by random choice to 
	make it smaller for KNN.
	"""

	index, subset_x, subset_y, temp, condensed = [], [], [], [], []

	index = list(range(len(training_x)))

	subset_x.append(training_x[0])
	subset_y.append(training_y[0])

	while sum(index):
		# need to keep track of indices
		nonzero_index = np.nonzero(index)
		remain_idx = r.choice(nonzero_index[0])
		index[remain_idx] = 0
		temp = []
		# append that index to temp set
		temp.append(training_x[remain_idx])
		# find our predicted value
		predicted_y = test_knn(subset_x, subset_y, temp, 3)
		if predicted_y[0] != training_y[remain_idx]:

			# example of being added to dataset in condensed
			point_example = training_x[remain_idx]
			
			subset_x.append(training_x[remain_idx])
			subset_y.append(training_y[remain_idx])
			condensed.append(remain_idx)

	print("Point being added to dataset: ", point_example)
	return condensed

def get_accuracy(testing_set, predictions):
	"""
	Gets the accuracy of the run.
	"""
	correct = 0
	for x in range(len(testing_set)):
		if testing_set[x] == predictions[x]:
			correct += 1
	return (correct / float(len(testing_set))) * 100.0



