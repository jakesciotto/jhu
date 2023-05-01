# -----------------------------------------------------------
# util.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import typing
import random
import math

# debugging options for pandas
"""
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
"""

def multi_winnow2(dataset, nb_features):
	"""
	For a dataset with more than 2 classes, this runs in a one vs. all and 
	aggregates the accuracy.
	"""

	train_total = 0
	test_total = 0
	# split up dataset
	attributes = dataset.iloc[:, :nb_features]
	classes = dataset.iloc[:, nb_features:]

	# encode both attributes and classes
	attributes_encoded = pd.get_dummies(attributes, columns = attributes.columns)
	classes_encoded = pd.get_dummies(classes, columns = classes.columns)

	# for each class, attach to attributes and run winnow2
	for column in classes_encoded.columns:
		df = pd.concat([attributes_encoded, classes_encoded[column]], axis = 1)
		""" 
		90% training
			- 2/3 training
			- 1/3 testing
		10% tuning
		"""
		training, tuning = split(df, .9)
		x_train, x_test = split(training, .67)
		target, trained, tuned = winnow2(x_train, tuning)

		train_acc = get_accuracy(target, trained)
		test_acc = get_accuracy(target, tuned)
		train_total += train_acc 
		test_total += test_acc

	total_train_acc = train_total / len(classes_encoded.columns)
	total_test_acc = test_total / len(classes_encoded.columns)

	return total_train_acc, total_test_acc

def winnow2(training, testing):
	"""
	Winnowing algorithm
	"""
	np_training_data = training.values
	np_testing_data  = testing.values

	"""
	Tuning parameters.
	"""
	THETA = .5
	ALPHA = 2.0

	instances = np_training_data.shape[0]
	columns = np_training_data.shape[1]
	attributes = columns - 2
	target_col = columns - 1

	weights = np.ones(attributes + 1)

	########################################################
	# TRAINING
	########################################################
	# new array with extra columns
	extra_training_cols = np.full((instances, 8), 99)

	# extra columns to the training set
	np_training_data = np.append(np_training_data, extra_training_cols, axis = 1)

	# array of floats
	np_training_data = np_training_data.astype(float)

	# build model by floating
	for row in range(0, instances):

		weighted_sum = 0

		# calculated weighted sum
		for col in range(1, attributes + 1):
			weighted_sum += (weights[col] * np_training_data[row, col])

		# record weighted sum right next to actual class column
		np_training_data[row, int(target_col) + 1] = weighted_sum

		predicted_class = 99

		"""
		A modification was made in the Winnow-2 algorithm to multiply 
		the Theta value by the number of attributes because the weighted sum
		was almost always greater than the threshold.
		"""
		if weighted_sum > THETA * attributes:
			predicted_class = 1
		else:
			predicted_class = 0

		np_training_data[row, int(target_col) + 2] = predicted_class

		actual_class = np_training_data[row, int(target_col)]

		# prediction outcomes
		tp = 0 
		fp = 0
		fn = 0
		tn = 0

		# updated outcomes of learner's prediction
		if predicted_class == 1 and actual_class == 1:
			tp = 1
		elif predicted_class == 1 and actual_class == 0:
			fp = 1
		elif predicted_class == 0 and actual_class == 1:
			fn = 1
		else:
			tn = 1

		# record
		np_training_data[row, int(target_col) + 3] = tp
		np_training_data[row, int(target_col) + 4] = fp
		np_training_data[row, int(target_col) + 5] = fn
		np_training_data[row, int(target_col) + 6] = tn

		promote = 0
		demote = 0

		# make the promotion or the demotion
		if fn == 1:
			promote = 1

		if fn == 1:
			demote = 1

		# record if either a promotion or demotion necessary
		np_training_data[row, int(target_col) + 7] = promote
		np_training_data[row, int(target_col) + 8] = demote

		# adjust weights by alpha
		if demote == 1:
			for col in range(1, attributes + 1):
				if (np_training_data[row, col] == 1):
					weights[col] /= ALPHA

		if promote == 1:
			for col in range(1, attributes + 1):
				if (np_training_data[row, col] == 1):
					weights[col] *= ALPHA
	########################################################
	# TESTING 
	########################################################
	instances = np_testing_data.shape[0]
	columns = np_testing_data.shape[1]
	attributes = columns - 2
	target_col = columns - 1

	# new array with extra columns
	extra_testing_cols = np.full((instances, 6), 99)

	# extra columns to the training set
	np_testing_data = np.append(np_testing_data, extra_testing_cols, axis = 1)

	# array of floats
	np_testing_data = np_testing_data.astype(float)

	for row in range(0, instances):

		weighted_sum = 0

		# calculated weighted sum
		for col in range(1, attributes + 1):
			weighted_sum += (weights[col] * np_testing_data[row, col])

		# record weighted sum right next to actual class column
		np_testing_data[row, int(target_col) + 1] = weighted_sum

		predicted_class = 99

		"""
		A modification was made in the Winnow-2 algorithm to multiply 
		the Theta value by the number of attributes because the weighted sum
		was almost always greater than the threshold.
		"""
		if weighted_sum > THETA * attributes:
			predicted_class = 1
		else:
			predicted_class = 0

		# update predicted class value
		np_testing_data[row, int(target_col) + 2] = predicted_class

		actual_class = np_testing_data[row, int(target_col)]

		# prediction outcomes
		tp = 0 
		fp = 0
		fn = 0
		tn = 0

		# updated outcomes of learner's prediction
		if predicted_class == 1 and actual_class == 1:
			tp = 1
		elif predicted_class == 1 and actual_class == 0:
			fp = 1
		elif predicted_class == 0 and actual_class == 1:
			fn = 1
		else:
			tn = 1

		# record
		np_testing_data[row, int(target_col) + 3] = tp
		np_testing_data[row, int(target_col) + 4] = fp
		np_testing_data[row, int(target_col) + 5] = fn
		np_testing_data[row, int(target_col) + 6] = tn

	return target_col, pd.DataFrame(data = np_training_data), pd.DataFrame(data = np_testing_data)      
              
def split(dataset, ratio):
	""" 
	Split the data as appropriate by ratio
	for Winnow-2.
	"""
	shuffled_df = dataset.sample(frac = 1)
	training_size = int(ratio * len(dataset))

	training = shuffled_df[:training_size]
	testing = shuffled_df[training_size:]

	return training, testing

def get_accuracy(target_col, dataset):
	"""
	Return accuracy based on target columns.
	"""

	# put accuracy function here 
	tp = dataset[target_col + 3].sum()
	fp = dataset[target_col + 4].sum()
	fn = dataset[target_col + 5].sum()
	tn = dataset[target_col + 6].sum()

	accuracy = (tp + tn)/(tp + tn + fp + fn)
	accuracy *= 100
	
	return accuracy
