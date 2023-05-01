# -----------------------------------------------------------
# util.py
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
import random

np.seterr(divide='ignore', invalid='ignore')


def split_xy(dataset):
	"""
	Splits the dataset into features and target class
	assuming that the last column in the dataset
	is 0s or 1s.
	"""
	X = dataset.iloc[:, :-1]
	y = (dataset.y != 0) * 1

	return X, y

def shuffle_split_data(X, y):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 90)

    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]

    return X_train, y_train, X_test, y_test

def normalize(dataset):
    """
    Performs minmax normalization on values.
    """
    dataset_normalized = stats.zscore(dataset, axis = 1, ddof = 1)
    return pd.DataFrame(data = dataset_normalized)

def one_vs_one_lr(dataset, k_folds, lr, num_iter, threshold):
    """
    For datasets that are binary classification problems, I'm using this
    method to run them from the `main.py` file.
    """
    total_accuracy = []

    for i in range(k_folds):

        X, y = split_xy(dataset)
        X = stats.zscore(X)
        X_train, y_train, X_test, y_test = shuffle_split_data(X, y)

        model = LogisticRegression(lr, num_iter)
        model.fit(np.array(X_train), y_train.values)
        preds = model.predict(X_test, threshold)
        accuracy = model.get_accuracy(preds, y_test.values)
        total_accuracy.append(accuracy)

    print("CV scores: \t", total_accuracy)
    print("Mean accuracy: \t", sum(total_accuracy) / len(total_accuracy))
    print()

def one_vs_one_ar(dataset, k_folds, eta, num_iter, random_state):
    """
    One vs. one for Adaline Regression.
    """
    total_accuracy = []

    for i in range(k_folds):

        X, y = split_xy(dataset)
        X = stats.zscore(X)
        X_train, y_train, X_test, y_test = shuffle_split_data(X, y)

        model = AdalineRegression(eta, num_iter, random_state)
        model.fit(X_train, y_train.values)
        preds = model.predict(X_test)
        accuracy = model.get_accuracy(preds, y_test.values)
        total_accuracy.append(accuracy)

    print("CV scores: \t", total_accuracy)
    print("Mean MSE: \t", sum(total_accuracy) / len(total_accuracy))
    print()

def one_vs_all_lr(dataset, k_folds, nb_features, lr, num_iter, threshold, verbose):
    """
    For datasets with more than one class, this method can run them one 
    vs. all and aggregate the accuracy.
    """

    total_accuracy = []

    for i in range(k_folds):

        accuracy = 0

        attributes = dataset.iloc[:, :nb_features]
        classes = dataset.iloc[:, nb_features:]

        classes_encoded = pd.get_dummies(classes, columns = classes.columns)

        for column in classes_encoded.columns:
            df = pd.concat([attributes, classes_encoded[column]], axis = 1)
            df.rename(columns = {df.columns[-1]: "y" }, inplace = True)

            X, y = split_xy(df)
            X = stats.zscore(X)
            X_train, y_train, X_test, y_test = shuffle_split_data(X, y)

            model = LogisticRegression(lr, num_iter)
            model.fit(np.array(X_train), y_train.values)
            preds = model.predict(X_test, threshold)
            accuracy += model.get_accuracy(preds, y_test.values)

        one_vs_all_accuracy = accuracy / len(classes_encoded.columns)
        total_accuracy.append(one_vs_all_accuracy)

    print("CV scores: \t", total_accuracy)
    print("Mean accuracy: \t", sum(total_accuracy) / len(total_accuracy))

    if verbose:
        print("Final weights: \t", model.theta)
        print("Final gradient: \t", model.final_gradient)

    print()

def one_vs_all_ar(dataset, k_folds, nb_features, eta, num_iter, random_state, verbose):
    """
    For datasets with more than one class, this method can run them one 
    vs. all and aggregate the accuracy.
    """
    total_accuracy = []

    for i in range(k_folds):
    
        accuracy = 0

        attributes = dataset.iloc[:, :nb_features]
        classes = dataset.iloc[:, nb_features:]

        classes_encoded = pd.get_dummies(classes, columns = classes.columns)

        for column in classes_encoded.columns:
            df = pd.concat([attributes, classes_encoded[column]], axis = 1)
            df.rename(columns = {df.columns[-1]: "y" }, inplace = True)

            X, y = split_xy(df)
            X = stats.zscore(X)
            X_train, y_train, X_test, y_test = shuffle_split_data(X, y)

            model = AdalineRegression(eta, num_iter, random_state)
            model.fit(X_train, y_train.values)
            preds = model.predict(X_test)
            accuracy += model.get_accuracy(preds, y_test.values)

        one_vs_all_accuracy = accuracy / len(classes_encoded.columns)
        total_accuracy.append(one_vs_all_accuracy)
    
    print("CV scores: \t", total_accuracy)
    print("Mean MSE: \t", sum(total_accuracy) / len(total_accuracy))

    if verbose:
        print("Final weights: \t", model.w_)
        print("Final gradient: \t", model.cost_)

    print()