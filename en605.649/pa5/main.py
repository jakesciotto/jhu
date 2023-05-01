# -----------------------------------------------------------
# main.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from nn import NeuralNetwork, Layer
from util import split_xy, shuffle_split_data, nn_classification, nn_regression, cor_selector

# import classification datasets
cancer = pd.read_csv("input/breast-cancer-wisconsin.data")
glass  = pd.read_csv("input/glass.data")
beans  = pd.read_csv("input/soybean-small.data")

# import regression datasets
abalone = pd.read_csv("input/abalone.data")
machine = pd.read_csv("input/machine.data")
fires   = pd.read_csv("input/forestfires.data")

###################################
# Data pre-processing
###################################

# Cancer dataset
# 
# Changes: missing values changed, categorical instances turned to numerical
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
#
# Changes: Columns renamed, id column dropped
glass.columns = ["id", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "y"]
glass.drop(["id"], axis = 1, inplace = True)

# Soybean dataset
#
# Changes: Columns renamed, categorical to numerical target class changes
beans.columns = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
	     		 "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19",
	     	     "x20", "x21", "x22", "x23", "x24", "x24", "x25", "x26", "x27",
	     	 	 "x28", "x29", "x30", "x31", "x32", "x33", "x34", "class"]

beans['class'] = beans['class'].astype('category')
beans['y'] = beans['class'].cat.codes
beans.drop(['class'], axis = 1, inplace = True)

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
fires = fires.rename(columns = {"area": "y"})

###################################
# Training the network
###################################

LEARNING_RATE = 0.1
EPOCHS = 10
K_FOLDS = 5

#frames = [cancer, glass, beans, abalone, machine, fires]
frames = [cancer]
names  = ["Cancer", "Glass", "Beans"]

for name, frame in zip(names, frames):
    nn_classification(name, frame, K_FOLDS, LEARNING_RATE, EPOCHS, 0)
    print()
    nn_classification(name, frame, K_FOLDS, LEARNING_RATE, EPOCHS, 1)
    print()
    nn_classification(name, frame, K_FOLDS, LEARNING_RATE, EPOCHS, 2)
    print()

"""
frames = [abalone, machine, fires]
names  = ["Abalone", "Machine", "Fires"]

for name, frame in zip(names, frames):
	nn_regression(name, frame, LEARNING_RATE, EPOCHS)
"""
