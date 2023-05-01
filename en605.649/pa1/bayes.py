# -----------------------------------------------------------
# bayes.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import math
import random

def split(dataset, ratio): 
	"""
	Split the data as appropriate by ratio
	for Naive Bayes.
	"""
	train_num = int(len(dataset) * ratio) 
	train = [] 
	test = list(dataset) 
	# make sure the index is shuffled
	while len(train) < train_num: 
		index = random.randrange(len(test)) 
		train.append(test.pop(index)) 
	return train, test 
  
def separate_class(dataset):
	"""
	Separate dataset by class.
	"""
	sep = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in sep):
			sep[class_value] = list()
		sep[class_value].append(vector)
	return sep
  
def mean(nums): 
	""" 
	Return mean.
	"""
	return sum(nums) / float(len(nums)) 
  
def std(nums): 
	"""
	Return standard deviation.
	"""
	avg = mean(nums) 
	"""
	This condition had to be added because of a division by zero error 
	when len(nums) == 1, we'd be dividing by zero.
	"""
	if len(nums) == 1:
		variance = 0
	else:
		variance = sum([pow(x - avg, 2) for x in nums]) / float(len(nums) - 1) 
	return math.sqrt(variance) 
  
def get_stats(dataset): 
	""" 
	Get stats for the dataset.
	"""
	stats = [(mean(x), std(x), len(x)) for x in zip(*dataset)] 
	del stats[-1] 
	return stats

def get_stats_class(dataset): 
	"""
	Get stats per class.
	"""
	info = dict()
	classes = separate_class(dataset)
	for class_value, instances in classes.items(): 
		info[class_value] = get_stats(instances) 
	return info 
  
def get_gaussian(x, mean, std): 
	"""
	Used to calculate the Gaussian probability density function.
	"""
	# to prevent a divide by zero error when standard deviation is zero
	if std == 0.0:
		if x == mean:
			return 1.0
		else:
			return 0.0
	expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2)))) 
	return (1 / (math.sqrt(2 * math.pi) * std)) * expo 
  
def get_probs(summaries, test): 
	"""
	Get probabilities per class.
	"""
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items(): 
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)): 
			mean, std, _ = class_summaries[i] 
			""" 
			Ordinarily the formula would be:
			(prior probability * density function) / total probability) 
			However, there are a fixed number of observations for each class so the 
			prior is fixed.
			"""
			probabilities[class_value] *= get_gaussian(test[i], mean, std) 
	return probabilities 
  
def predict(info, test): 
	"""
	Make the prediction - highest probability is best.
	"""
	probabilities = get_probs(info, test) 
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items(): 
		if best_label is None or probability > best_prob: 
			best_prob = probability 
			best_label = class_value 
	return best_label 
  
def naive_bayes(training, testing): 
	"""
	Naive Bayes function.
	"""
	info = get_stats_class(training)
	predictions = []
	for i in range(len(testing)): 
		result = predict(info, testing[i]) 
		predictions.append(result) 
	return predictions 

def cross_val(dataset, folds):
	""" 
	Perform cross-validation split of dataset before Naive Bayes.
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
	return split_data

def evaluate(dataset, n_folds):
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
			copy[-1] = None

		# perform naive bayes on each fold
		preds = naive_bayes(training, testing)
		actual = [row[-1] for row in fold]
		accuracy = get_cv_accuracy(actual, preds)
		scores.append(accuracy)
	return scores

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