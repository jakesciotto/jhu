# -----------------------------------------------------------
# ar.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

import numpy as np

class AdalineRegression:

	def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):

		self.eta = eta
		self.n_iter = n_iter
		self.shuffle = shuffle
		self.random_state = random_state

		if random_state:
			np.random.seed(random_state)

	def fit(self, X, y):
		""" 
		Fits training data. Both inputs are array-like.
		"""

		self._initialize_weights(X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X, y)

			cost = []

			for xi, target in zip(X, y):
				cost.append(self._update_weights(xi, target))

			avg_cost = sum(cost) / len(y)
			self.cost_.append(avg_cost)

		return self

	def partial_fit(self, X, y):
		"""
		Fits training data without re-initializing the weights.
		"""
		if not self.w_initialized:
			self._initialize_weights(X.shape[1])

		if y.ravel().shape[0] > 1:
			for xi, target in zip(X, y):
				self._update_weights(xi, target)

		else:
			self._update_weights(X, y)

		return self

	def _shuffle(self, X, y):
		"""
		Shuffles training data between epochs. 
		"""
		r = np.random.permutation(len(y))
		return X[r], y[r]

	def _initialize_weights(self, m):
		"""
		Initialize weights to 0.
		"""
		self.w_ = np.zeros(1 + m)
		self.w_initialized = True

	def _update_weights(self, xi, target):
		"""
		Applies Adaline learning rule to update the weights.
		"""
		output = self.net_input(xi)
		error = target - output
		self.w_[1:] += self.eta * xi.dot(error)
		self.w_[0] += self.eta * error
		cost = 0.5 * (error ** 2)

		return cost

	def net_input(self, X):
		"""
		Calculates net input.
		"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):

		""" Compute linear activation """
		return self.net_input(X)

	def predict(self, X):

		""" Return class label after the unit step """

		return np.where(self.activation(X) >= 0.0, 1, -1)

	def get_accuracy(self, predicted, y):
		"""
		Return the accuracy of the predictions.
		"""
		sum_error = 0.0
		for i in range(len(y)):
			prediction_error = predicted[i] - y[i]
			sum_error += (prediction_error ** 2)
		mean_error = sum_error / float(len(y))
		return mean_error