# -----------------------------------------------------------
# cart.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
#
# Modified from
# https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
# -----------------------------------------------------------

from datetime import datetime
import pandas as pd
import numpy as np
import util, dt

class CART:
  
    def fit(self, X, y, min_leaf = 1):
        """
        Fits the testing y to our model.
        """
        self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf)
        return self

    def predict(self, X):
        """
        Function to make predictions.
        """
        return self.dtree.predict(X.values)

    def get_r2(self, x, y):
        """
        R^2 score for decision tree regression.
        """
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = 1 - (sum((y - (slope * x + intercept)) ** 2) / ((len(y) - 1) * np.var(y, ddof = 1)))
        return r_squared

class Node:
    # every time this code gets called, we want to have this executed
    def __init__(self, x, y, indexes, min_leaf = 1):
        self.x = x 
        self.y = y
        self.indexes = indexes 
        self.min_leaf = min_leaf
        self.row_count = len(indexes)
        self.col_count = x.shape[1]
        self.val = np.mean(y[indexes])
        self.score = float('inf')
        self.find_var_split()
        
    def find_var_split(self):
        """
        For every variable, we want to find the best place to split. 
        """
        for c in range(self.col_count): 
            self.find_better_split(c)

        # return once we have hit a leaf
        if self.is_leaf: 
            return

        x = self.split_col

        # nonzero numpy method finds indices in an array where a condition is true
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]

        # every node, including the root node, is assigned a predicted class
        self.lhs = Node(self.x, self.y, self.indexes[lhs], self.min_leaf)
        self.rhs = Node(self.x, self.y, self.indexes[rhs], self.min_leaf)
        
    def find_better_split(self, var_index):
        """
        This is where the split criterion is 
        being formulated. If the weighted sum
        is less than our right or left child, 
        we continue until we find the better split.
        """
        x = self.x.values[self.indexes, var_index]

        for r in range(self.row_count):
            """
            For every row, the right children are greater than the current stored value 
            at the index. Left children are the opposite.
            """
            lhs = x <= x[r]
            rhs = x > x[r]

            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf: 
                continue

            curr_score = self.find_score(lhs, rhs)
            if curr_score < self.score: 
                self.var_index = var_index
                self.score = curr_score
                self.split = x[r]
                
    def find_score(self, lhs, rhs):
        """
        To find the score, we can use standard deviation to determine the 
        homogeneity of the sample.
        """
        y = self.y[self.indexes]
        lhs_std = y[lhs].std()
        rhs_std = y[rhs].std()
        return lhs_std * lhs.sum() + rhs_std * rhs.sum()
                
    @property
    def split_col(self): 
        return self.x.values[self.indexes, self.var_index]
                
    @property
    def is_leaf(self): 
        """
        Returns true if our score is infinity, which designates it as a leaf.
        """
        return self.score == float('inf')                

    def predict(self, x):
        """
        Used to create a row of predictions based on an instance x_i
        """
        return np.array([self.predict_row(x_i) for x_i in x])

    def predict_row(self, x_i):
        """
        Helper function for predict that performs prediction for each instance in row.
        """
        if self.is_leaf: 
            return self.val
        node = self.lhs if x_i[self.var_index] <= self.split else self.rhs
        return node.predict_row(x_i)



