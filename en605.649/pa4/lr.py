# -----------------------------------------------------------
# lr.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

import pandas as pd
import numpy as np 

class LogisticRegression:

	def __init__(self, lr = 0.01, num_iter = 100000, fit_intercept = True, verbose = False, final_gradient = 0):
		"""
		Default parameters for the LogisticRegression class.
		"""
		self.lr = lr
		self.num_iter = num_iter
		self.fit_intercept = fit_intercept
		self.verbose = verbose
		self.final_gradient = final_gradient

	def __add_intercept(self, X):
		"""
		This method will add an intercept column to the dataset.
		"""
		intercept = np.ones((X.shape[0], 1))
		return np.concatenate((intercept, X), axis = 1)
    
	def __sigmoid(self, z):
		"""
		Represents our sigmoid function.
		"""
		return 1 / (1 + np.exp(-z))

	def _loss_function(self, h, y):
		"""
		Calculation of the loss.
		"""
		return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

	def fit(self, X, y):
		""" 
			theta 	 : 1D vector containing weights for each attribute
			x 	  	 : vector containing the values of each attribute for a specific instance 
			h 	  	 : probability that an instance is member of a positive class 
			gradient : result of the dot product of the class instances and the error (diff between h and the class)
		"""
		if self.fit_intercept:
			X = self.__add_intercept(X)
        
        # weights initialization
		self.theta = np.zeros(X.shape[1])
        
		for i in range(self.num_iter):
			z = np.dot(X, self.theta)
			h = self.__sigmoid(z)

			# calculate gradient
			gradient = np.dot(X.T, (h - y)) / y.size

			# update the weight vector with new gradient
			self.theta -= self.lr * gradient
            
			if(self.verbose == True and i % 10000 == 0):
				z = np.dot(X, self.theta)
				h = self.__sigmoid(z)
				print(f'loss: {self._loss_function(h, y)} \t')

			self.final_gradient = gradient
    
	def predict_prob(self, X):
		"""
		Calling this function calculates the probability that some 
		input X belongs to class 1.
		"""
		if self.fit_intercept:
			X = self.__add_intercept(X)
    
		return self.__sigmoid(np.dot(X, self.theta))
    
	def predict(self, X, threshold):
		"""
		Making predictions based on our critical threshold. What this means
		is that all probabilities greater than our threshold will be classified
		in one class.
		"""
		return self.predict_prob(X) >= threshold

	def get_accuracy(self, preds, y):
		"""
		Return the accuracy of the predictions.
		"""
		correct = 0
		for i in range(len(preds)): 
			if preds[i] == y[i]: 
				correct += 1
		return (correct / float(len(preds))) * 100.0