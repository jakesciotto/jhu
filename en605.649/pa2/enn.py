# -----------------------------------------------------------
# enn.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import operator
import copy
import math, knn

def get_neighbors(training, testing_row, nb_neighbors):
	"""
	Get nearest neighbors for KNN classiication.
	"""
	distances = list()
	for training_row in training:
		d = knn.euclidean_distance(testing_row, training_row)
		distances.append((training_row, d))
	distances.sort(key = lambda x: x[1])
	neighbors = list()
	for i in range(nb_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def euclidean_distance(row1, row2):
	"""
	Calculation of Euclidean distance for KNN.
	"""
	distance = 0.0
	for i in range(len(row1) - 1):
		distance += (row1[i] - row2[i]) ** 2
	return math.sqrt(distance)


def predict(training, testing_row, nb_neighbors):
	"""
	Make a classification prediction.
	"""
	neighbors = get_neighbors(training, testing_row, nb_neighbors)
	result = [row[-1] for row in neighbors]
	return max(set(result), key = result.count)

def edit(training_x, testing_x, nb_neighbors):
	"""
	Function to edit out values based on incorrect instances in 
	KNN.
	"""

	edited_data = copy.deepcopy(training_x)
	df = edited_data.values

	improving = True
	cur_perf = 0
	loop = 0
	i = 0

	# while the algorithm is improving, continue
	while improving:
		for row in df:
			correct = testing_x[i]
			predicted = predict(df, row, nb_neighbors)
			
			# start making new dataset of columns
			edited_data.loc[i, "correct"] = float(correct)
			edited_data.loc[i, "predicted"] = predicted

			past_perf = cur_perf
			cur_perf = get_accuracy(df, row, nb_neighbors)
			i += 1

			# stop when we stop improving
			if cur_perf < past_perf:
				improving = False	

	edited_data["result"] = np.where(edited_data["correct"] == edited_data["predicted"], "True", "False")
	print(edited_data)
	final_df = edited_data[edited_data["result"] == "True"]
	return final_df

def get_accuracy(training_x, testing_x, nb_neighbors): 
	""" 
	Determine accuracy rate.
	"""
	correct = 0
	for i in range(len(testing_x)):
		actual = testing_x[i]
		predicted = predict(training_x, testing_x, nb_neighbors)
		if predicted == actual:
			correct += 1

	return correct / len(testing_x)