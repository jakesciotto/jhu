# -----------------------------------------------------------
# main.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import winnow, bayes

# individually assigning dataframes
cancer = pd.read_csv("input/breast-cancer-wisconsin.data")
glass  = pd.read_csv("input/glass.data")
votes  = pd.read_csv("input/house-votes-84.data")
iris   = pd.read_csv("input/iris.data")
beans  = pd.read_csv("input/soybean-small.data")

# constant
LINE = "----------------------------------------------------------"

#######################################
# Data pre-processing
#
# Each of these datasets has their own set of issues, and 
# I will explore them one by one.
#######################################

# cancer dataset
cancer.columns = ["id", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "y"]

# replace missing values, convert 4s and 2s to 1s and 0s
cancer.replace("?", np.nan, inplace = True)
cancer.fillna(cancer.median(), inplace = True)
# drop id column
cancer.drop(['id'], axis = 1, inplace = True)
# encode malignant and benign values
cancer['y'].replace(4, 1, inplace = True)
cancer['y'].replace(2, 0, inplace = True)
cancer = cancer.astype(int)

# glass dataset
glass.columns = ["id", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "y"]

# drop id column
glass.drop(['id'], axis = 1, inplace = True)

# voting dataset
votes.columns = ["class", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
				 "x11", "x12", "x13", "x14", "x15", "x16"]

# replace ? with 2 since it means abstain
votes.replace("?", 2, inplace = True)
votes.replace("y", 1, inplace = True)
votes.replace("n", 0, inplace = True)

# move republican/democrat column to the end and set to numerical
votes['y'] = votes['class'].apply(lambda x: 1 if x == 'republican' else 0)
votes.drop(['class'], axis = 1, inplace = True)

# iris dataset
iris.columns = ["x1", "x2", "x3", "x4", "class"]

# turn categorical labels into numerical
iris['class'] = iris['class'].astype('category')
# encode the class values
iris['y'] = iris['class'].cat.codes
iris.drop(['class'], axis = 1, inplace = True)
iris = iris.astype(float)

# soybean dataset
beans.columns = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
	     		 "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19",
	     	     "x20", "x21", "x22", "x23", "x24", "x24", "x25", "x26", "x27",
	     	 	 "x28", "x29", "x30", "x31", "x32", "x33", "x34", "class"]

# turn categorical labels into numerical
beans['class'] = beans['class'].astype('category')
# encode class values
beans['y'] = beans['class'].cat.codes
beans.drop(['class'], axis = 1, inplace = True)

# put frames together
names  = ["Cancer", "Glass", "Votes", "Iris", "Soybeans"]
frames = [cancer, glass, votes, iris, beans]

"""
This is the section of the code where I would:
	- Look for outliers
	- Check and make sure features are independent
	- Perform feature ranking or reduce dimensionality
"""
for frame in frames:
	corr_matrix = frame.corr()

##########################
# Naive Bayes
##########################

# training on tuning set
print("Naive Bayes")
print("Training on 10% tuning set")
print(LINE)
for name, frame in zip(names, frames):
	training, tuning = bayes.split(list(frame.values), .9)
	x_train, x_test = bayes.split(training, .67)
	predictions = bayes.naive_bayes(x_train, tuning)
	accuracy = bayes.get_accuracy(tuning, predictions)
	print("Dataset: \t", name) 
	print("Accuracy: \t", accuracy)
	print("")

print("")

# without tuning parameters
print("Naive Bayes")
print("Cross validation folds: 5")
print(LINE)
# perform cross validation
for name, frame in zip(names, frames):
	training, testing = bayes.split(list(frame.values), .9)
	x_train, x_test = bayes.split(list(frame.values), .67)
	cv_scores = bayes.evaluate(x_train, 5)
	print("Dataset: \t", name)
	print("Scores: \t", cv_scores)
	print("Mean accuracy: \t", sum(cv_scores) / float(len(cv_scores)))
	print("")

print("")

##########################
# Winnow-2
##########################
print("Winnow-2:")
print(LINE)

# encoding the cancer dataset before winnow-2
for column in range(len(cancer.columns) - 1):
	cancer.iloc[:, column] = cancer.iloc[:, column].apply(lambda x: 1 if x >= 6 else 0)

names = ["Cancer", "Votes"]
dfs = [cancer, votes]

for name, df in zip(names, dfs):
	""" 
	90% training
		- 2/3 training
		- 1/3 testing
	10% tuning
	"""
	training, tuning = winnow.split(df, .9)
	x_train, x_test = winnow.split(training, .67)
	target, trained, tuned = winnow.winnow2(x_train, tuning)

	# get accuracy scores 
	train_acc = winnow.get_accuracy(target, trained)
	test_acc = winnow.get_accuracy(target, tuned)
	print("Dataset: \t\t", name)
	print("Training accuracy: \t", train_acc)
	print("Testing accuracy: \t", test_acc)
	print("")

"""
3 datasets have multi-class situations that need to be run 
one vs. all. Multi_winnow2 function performs this action.
"""
names = ["Glass", "Iris", "Soybeans"]
dfs_ = [glass, iris, beans]

for name, df in zip(names, dfs_):
	train_result, test_result = winnow.multi_winnow2(df, len(df.columns) - 1)
	print("Dataset: \t\t", name)
	print("Training accuracy: \t", train_result)
	print("Testing accuracy: \t", test_result)
	print("")