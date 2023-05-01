# -----------------------------------------------------------
# main.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

from lr import LogisticRegression
from ar import AdalineRegression
from scipy import stats
import pandas as pd
import numpy as np
import util

# individually assigning dataframes
cancer = pd.read_csv("input/breast-cancer-wisconsin.data")
glass  = pd.read_csv("input/glass.data")
votes  = pd.read_csv("input/house-votes-84.data")
iris   = pd.read_csv("input/iris.data")
beans  = pd.read_csv("input/soybean-small.data")

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

# Glass dataset
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

##########################
# Logistic Regression
##########################

names = ["Cancer", "Glass", "Votes", "Iris", "Beans"]
frames = [cancer, glass, votes, iris, beans]

K_FOLDS = 5
L_RATE = 10
ITERATIONS = 5000
THRESHOLD = .5

print(" ***** Beginning Logistic Regression ***** ")

for name, frame in zip(names, frames):

	if name == "Cancer" or name == "Votes":
		print("Dataset: \t", name)
		util.one_vs_one_lr(frame, K_FOLDS, L_RATE, ITERATIONS, THRESHOLD)
	else:
		print("Dataset: \t", name)
		if name == "Iris":
			util.one_vs_all_lr(frame, K_FOLDS, len(frame.columns) - 1, L_RATE, ITERATIONS, THRESHOLD, verbose = True)
		else:
			util.one_vs_all_lr(frame, K_FOLDS, len(frame.columns) - 1, L_RATE, ITERATIONS, THRESHOLD, verbose = False)

##########################
# Adaline Regression
##########################

K_FOLDS = 5
ETA = .01
ITERATIONS = 10
RANDOM_STATE = 1

print("\n ***** Beginning Adaline Regression ***** ")

for name, frame in zip(names, frames):
	if name == "Cancer" or name == "Votes":
		print("Dataset: \t", name)
		util.one_vs_one_ar(frame, K_FOLDS, ETA, ITERATIONS, RANDOM_STATE)
	else:
		print("Dataset: \t", name)
		if name == "Iris":
			util.one_vs_all_ar(frame, K_FOLDS, len(frame.columns) - 1, ETA, ITERATIONS, RANDOM_STATE, verbose = True)
		else:
			util.one_vs_all_ar(frame, K_FOLDS, len(frame.columns) - 1, ETA, ITERATIONS, RANDOM_STATE, verbose = False)