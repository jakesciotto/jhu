# -----------------------------------------------------------
# util.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

import numpy as np
import pandas as pd
import random
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

"""
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
"""

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

def shuffle_split_data(X, y):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 90)

    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]

    return X_train, y_train, X_test, y_test

def feature_ranking(dataset):
	"""
	Performs feature ranking for a dataset based on
	Bhattacharyya distance.
	"""
	print(dataset)
	nb_features = dataset.shape[1] - 2
	array = dataset.values
	names = dataset.columns[0:nb_features]

	X = array[:, 0:nb_features]
	Y = array[:, dataset.shape[1] - 1]

	bhat_dist = [0] * nb_features

	feature_selection = ["jan", "feb"]

	for i, name in enumerate(names):
		X1 = np.array(X[:, i], dtype=np.float64)[Y == feature_selection[0]]
		X2 = np.array(X[:, i], dtype=np.float64)[Y == feature_selection[1]]
		bhat_dist[i] = bhattacharyya(X1, X2)

	for n, d in sorted(zip(names, bhat_dist)):
		distance = str("Bhattacharyya distance for: ") + str(n) + " " + str(d)
		print(distance)

def bhattacharyya(X1, X2):

	"""
	Bhattacharyya distance method

	Theoretical calculation using formula (under 
	normal distribution)

	Db(p,q) = (ln((v_p / v_q) + (v_q + v_p) + 2) / 4) +
				(((u_p - u_q)^2 / (v_p + v_q)) / 4)
	"""

	# calculate standard deviation
	s1 = np.std(X1)
	s2 = np.std(X2)

	# calculate mean
	m1 = np.mean(X1)
	m2 = np.mean(X2)

	# calculate variance
	v1 = s1 ** 2
	v2 = s2 ** 2

	# use formula for theortical distribution
	bdist = np.log(((v1 / v2 + v2 / v1) + 2) / 4) / 4 + (((m1 - m2) ** 2 / (v1 + v2)) / 4)

	return bdist