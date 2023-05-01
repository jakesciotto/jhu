# -----------------------------------------------------------
# main.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

from collections import Counter
from datetime import datetime
from cart import CART
from math import log
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from id3 import Node
import random
import json
import util
import id3
import dt

# constants
N_FOLDS = 5
MAX_DEPTH = 1
MIN_SIZE = 3

# import classification datasets
cancer = "input/cancer.data"
segs   = "input/segmentation.data"
cars   = "input/cars.data"

# import regression datasets
abalone = pd.read_csv("input/abalone.data")
machine = pd.read_csv("input/machine.data")
fires   = pd.read_csv("input/forestfires.data")

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

total_runtime = datetime.now()

# ID3 ALGORITHM
# ---------------------------------
print("***** Starting ID3 Algorithm *****")
print("-----------------------------------")

files = [cancer, segs, cars]
names = ["Cancer", "Segmentation", "Cars"]

for name, file in zip(names, files):

	pruned_scores, unpruned_scores = [], []

	dataset = util.parse_file(file)

	random.shuffle(dataset)

	base = id3.find_mode(dataset)

	# 90/10 split, 10% kept for validation
	validation = dataset[: 1 * len(dataset) // 10]
	dataset = dataset[1 * len(dataset) // 10 : len(dataset)]

	# get folds and put them together in a rotated fashion
	fold0, fold1, fold2, fold3, fold4 = util.get_five_folds(dataset)
	training, testing = util.put_folds_together(fold0, fold1, fold2, fold3, fold4)

	print("Dataset: ", name)

	for i in range(5):

		start = datetime.now()

		x_train = training[i]
		y_test  = testing[i]

		# pruned
		tree = id3.id3(x_train, base)

		if (name == "Cars"):
			id3.print_tree(tree)
		id3.prune(tree, validation)
		accuracy = id3.get_accuracy(tree, y_test)
		pruned_scores.append(accuracy)

		print("Classification accuracy (pruned): ", sum(pruned_scores) / len(pruned_scores))
	print("Running time: ", datetime.now() - start)
	print()

	for i in range(5):
		start = datetime.now()

		x_train = training[i]
		y_test  = testing[i]

		# unpruned
		tree = id3.id3(x_train, base)
		accuracy = id3.get_accuracy(tree, y_test)
		unpruned_scores.append(accuracy)

		print("Classification accuracy (unpruned): ", sum(unpruned_scores) / len(unpruned_scores))
	print("Running time: ", datetime.now() - start)
	print()

	print("Average classification accuracy (pruned): %.2f%%" % ((sum(pruned_scores) / 5) * 100))
	print("Average classification accuracy (unpruned): %.2f%%" % ((sum(unpruned_scores) / 5) * 100))
	print()
# CART ALGORITHM
# ---------------------------------

_abalone = util.choose_at_random(abalone, 250)
_fires   = util.choose_at_random(fires, 250)

names  = ["Abalone", "Machine", "Fires"]
frames = [_abalone, machine, _fires]

print("***** Starting CART Algorithm *****")
print("-----------------------------------")

for name, frame in zip(names, frames):

	training, testing = util.train_test_split(frame.values, .9)

	start = datetime.now()

	print("Dataset: ", name)
	scores = dt.evaluate_single(frame, dt.decision_tree, False, MAX_DEPTH, MIN_SIZE)
	print('Scores: \t', scores)
	print("MSE: \t", (sum(scores) / float(len(scores))))
	print("Running time: \t", datetime.now() - start)
	print()
	
	start = datetime.now()

	print("Dataset: ", name)
	print('Implementing k-cross validation')
	cv_scores = dt.evaluate(training, dt.decision_tree, N_FOLDS, False, MAX_DEPTH, MIN_SIZE)
	print("Scores: \t", cv_scores)
	print("Mean MSE \t", (sum(cv_scores) / float(len(cv_scores))))
	print("Running time: \t", datetime.now() - start)
	print()

# CART ALGORITHM WITH STOPPING
# ----------------------------------

_abalone = util.choose_at_random(abalone, 250).reset_index(drop = True)
_fires   = util.choose_at_random(fires, 250).reset_index(drop = True)

names  = ["Abalone", "Machine", "Fires"]
frames = [_abalone, machine, _fires]

print("***** Starting CART Algorithm *****")
print("     ***** (with tuning) *****     ")
print("-----------------------------------")

for name, frame in zip(names, frames):
	
	X = frame[frame.columns[:-1]]
	y = frame[frame.columns[-1]]

	start = datetime.now()

	reg = CART().fit(X, y)
	preds = reg.predict(X)
	r2 = reg.get_r2(preds, y)

	accuracy = dt.get_accuracy(y, preds)
	print("Dataset: ", name)
	print("MSE: ", accuracy)
	print("Running time: ", datetime.now() - start)
	print("R2: ", r2)
	print()
print("TOTAL RUNTIME: ", datetime.now() - total_runtime)

# TESTING SECTION
# ----------------------------------
"""
To demonstrate certain aspects of my algorithms, I have used small (~20 instances) of each
dataset and have print statements for each one.
"""
print("\n\n****** Starting Classification Test ******")
print("-----------------------------------")
cancer = util.parse_file("input/cancer.data")
cancer = cancer[1:20]
cancer_ = cancer[1:40]
_abalone = util.choose_at_random(abalone, 20).reset_index(drop = True)

# classification test
validation = cancer[: 1 * len(cancer) // 10]
cancer = cancer[1 * len(cancer) // 10 : len(cancer)]

# unpruned
print("Unpruned results")
base = id3.find_mode(cancer, True)
tree = id3.id3(cancer, base, True)
id3.print_tree(tree)
accuracy = id3.get_accuracy(tree, validation, True)
print()

# pruned 
print("Pruned results")
base = id3.find_mode(cancer, True)
base_ = id3.find_mode(cancer_, False)
tree = id3.id3(cancer, base, True)
tree_ = id3.id3(cancer_, base_, True)
id3.prune(tree, validation, True)
id3.print_tree(tree)
accuracy = id3.get_accuracy(tree, validation, True)
print()

# regression tree test
print("\n\n****** Starting Regression Test ******")
print("-----------------------------------")
scores = dt.evaluate_single(_abalone, dt.decision_tree, True, 5, 3)
print('Scores: \t', scores)
print("MSE without stopping: \t", (sum(scores) / float(len(scores))))

X = _abalone[_abalone.columns[:-1]]
y = _abalone[_abalone.columns[-1]]

reg = CART().fit(X, y)
preds = reg.predict(X)
r2 = reg.get_r2(preds, y)
id3.print_tree(tree_)
accuracy = dt.get_accuracy(y, preds)
print("MSE with stopping: ", accuracy)
print("R2: ", r2)

