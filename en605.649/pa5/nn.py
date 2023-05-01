# -----------------------------------------------------------
# nn.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
#
# Modified from source: 
# https://github.com/zHaytam/FlexibleNeuralNetFromScratch/blob/master/neural_network.py
# -----------------------------------------------------------

import numpy as np

class Layer:
	"""
	Represents a layer (hidden or output) in our neural network.
	"""
	def __init__(self, n_input, n_neurons, activation = None, weights = None, bias = None):
		"""
		Initial parameters for the Layer class. Bias is not used.
		"""
		self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons)
		self.activation = activation
		self.bias = bias if bias is not None else np.random.randn(n_neurons)
		self.last_activation = None
		self.error = None
		self.delta = None

	def activate(self, x):
		"""
		Calculates the dot product of this layer.
		"""
		r = np.dot(x, self.weights) + self.bias
		self.last_activation = self._apply_activation(r)
		return self.last_activation

	def _apply_activation(self, r):
		"""
		Applies the chosen activation function (if any).
		"""
		# none
		if self.activation is None:
			return r

		# tanh
		if self.activation == 'tanh':
			return np.tanh(r)

		# sigmoid
		if self.activation == 'sigmoid':
			return 1.0 / (1.0 + np.exp(-np.double(r)))

		# softmax
		if self.activation == 'softmax':
			e = np.exp(r - np.max(r))
			return e / e.sum(axis = 0)

		return r

	def apply_activation_derivative(self, r):
		"""
		Applies the derivative of the activation function (if any).
		"""

		if self.activation is None:
			return r

		if self.activation == 'tanh':
			return 1 - r ** 2

		if self.activation == 'sigmoid':
			return r * (1 - r)

		return r

class NeuralNetwork:
	"""
	Represents a neural network.
	"""
	def __init__(self):
		self._layers = []

		# added for testing
		self._gradient = None
		self._last_activation = None
		self._last_input = None

	def add_layer(self, layer):
		"""
		Adds a layer to the neural network.
		"""
		self._layers.append(layer)

	def feed_forward(self, X):
		"""
		Feed forward the input through the layers.
		"""
		for layer in self._layers:
			X = layer.activate(X)

		return X

	def predict(self, X):
		"""
		Predicts a class (or classes).
		"""
		ff = self.feed_forward(X)
		return ff

	def backprop(self, X, y, learning_rate):
		"""
		Performs the backward propagation algorithm and updates the layers weights.
		"""
		# get output from feed_forward
		output = self.feed_forward(X)

		# loop over layers backwards
		for i in reversed(range(len(self._layers))):
			layer = self._layers[i]

			# if output layer
			if layer == self._layers[-1]:
				layer.error = y - output
				# output is last application derivative
				layer.delta = layer.error * layer.apply_activation_derivative(output)
			else:
				next_layer = self._layers[i + 1]
				layer.error = np.dot(next_layer.weights, next_layer.delta)
				layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

		# update weights
		for i in range(len(self._layers)):
			layer = self._layers[i]
			# input is either the output from last layer or X
			input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
			layer.weights += layer.delta * input_to_use.T * learning_rate
			# added for testing
			self._gradient = layer.delta
			self._last_activation = layer.last_activation
			self._last_input = input_to_use

	def train(self, X, y, learning_rate, max_epochs):
		"""
		Trains the neural network using backprop to a max number of epochs.
		"""
		mses = []

		for i in range(max_epochs):
			for j in range(len(X)):
				self.backprop(X[j], y[j], learning_rate)
				mse = np.mean(np.square(y - self.feed_forward(X)))
				mses.append(mse)

		return mses

	def get_final_info(self):
		"""
		Displays final information for testing purposes.
		"""
		print("Gradient calculation: \t", self._gradient)
		print("Last activation function calculation: \t", self._last_activation)
		print("Last input used: \t", self._last_input)

	@staticmethod
	def get_r2(x, y):
		"""
		R^2 score for regression.
		"""
		correlation_matrix = np.corrcoef(x, y)
		correlation_xy = correlation_matrix[0, 1]
		return correlation_xy ** 2

	@staticmethod
	def mse_metric(y_pred, y_true):
		"""
		Calculates the MSE for regression.
		"""
		sum_error = 0.0
		for i in range(len(y_true)):
			prediction_error = y_pred[i] - y_true[i]
			sum_error += (prediction_error ** 2)
		mean_error = sum_error / float(len(y_true))
		return mean_error

	@staticmethod
	def accuracy(y_pred, y_true):
		"""
		Calculates the accuracy between the predicted labels and true labels.
		"""
		
		correct = 0

		for i in range(len(y_true)):
			if np.round(y_pred[i], 1) == y_true[i]:
				correct += 1
		return correct / float(len(y_true)) * 100.0