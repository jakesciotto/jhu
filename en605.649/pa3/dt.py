# -----------------------------------------------------------
# dt.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

import random
import util
import math

def get_accuracy(actual, predicted):
	"""
	Returns accuracy.
	"""
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return mean_error

def evaluate_single(dataset, algorithm, flag, *args):
	"""
	Evaluate the algorithm using a single split.
	"""
	training, testing = util.train_test_split(dataset.values, .9)
	scores = list()
	predicted = algorithm(training, testing, flag, *args)
	actual = [row[-1] for row in testing]
	accuracy = get_accuracy(actual, predicted)
	scores.append(accuracy)
	return scores
 
def evaluate(dataset, algorithm, n_folds, *args):
	"""
	Evaluate the decision tree.
	"""
	folds = util.cross_val(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		# I think the remove method is a Python2 error
		try:
			train_set.remove(fold)
		except:
			pass
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = get_accuracy(actual, predicted)
		scores.append(accuracy)
	return scores

def test_split(index, value, dataset):
	"""
	Testing where the split should be.
	"""
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

def idx_gini(groups, classes):
	"""
	Gini impurity is a the way to calculate the homoegeneity of samples.
	"""
	instances = float(sum([len(group) for group in groups]))
	gini = 0.0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0.0
		for val_class in classes:
			p = [row[-1] for row in group].count(val_class) / size
			score += p * p
			gini += (1.0 - score) * (size / instances)
	return gini

def get_split(dataset):
	"""
	Determine where we want to split based on the gini impurity.
	"""
	class_values = list(set(row[-1] for row in dataset))

	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0]) - 1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = idx_gini(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups

	return {'index': b_index, 'value': b_value, 'groups': b_groups}

def create_terminal_node(group, flag):
	"""
	Create a terminal node when we get to a leaf.
	"""
	result = [row[-1] for row in group]
	if flag:
		print("Prediction at leaf: ", max(set(result), key = result.count))
	return max(set(result), key = result.count)
 
def split_nodes(node, flag, max_depth, min_size, depth):
	"""
	Logic to create a terminal node or not. Gini impurity measures the divergences 
	between the probability distributions of the target attribute's values and splits a 
	node such that it gives the least amount of impurity.
	"""
	left, right = node['groups']

	del(node['groups'])

	# if we hit a terminal node, return
	if not left or not right:
		node['left'] = node['right'] = create_terminal_node(left + right, flag)
		return

	# if we are at max depth, return
	if depth >= max_depth:
		node['left'], node['right'] = create_terminal_node(left, flag), create_terminal_node(right, flag)
		return

	# look at left child
	if len(left) <= min_size:
		node['left'] = create_terminal_node(left, flag)
	else:
		node['left'] = get_split(left)
		split_nodes(node['left'], flag, max_depth, min_size, depth + 1)
	# otherwise, right child
	if len(right) <= min_size:
		node['right'] = create_terminal_node(right, flag)
	else:
		node['right'] = get_split(right)
		split_nodes(node['right'], flag, max_depth, min_size, depth + 1)
 
def build_tree(train, flag, max_depth, min_size):
	"""
	Driver that builds the decision tree.
	"""
	root = get_split(train)
	split_nodes(root, flag, max_depth, min_size, 1)
	return root

def predict(node, row):
	"""
	We want to get all the way to a leaf node, 
	so use recursion until we reach a leaf node.
	"""
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def decision_tree(train, test, flag, max_depth, min_size):
	"""
	Helper method to call the build_tree method and build
	predictions.
	"""
	tree = build_tree(train, flag, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	if flag:
		print("Predictions: ", predictions)
	return predictions 