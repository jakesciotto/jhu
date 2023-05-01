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
from scipy import stats
from datetime import datetime
from nn import NeuralNetwork, Layer

np.seterr(divide='ignore', invalid='ignore')

def split_xy(dataset):
	"""
	Splits the dataset into features and target class
	assuming that the last column in the dataset
	is 0s or 1s.
	"""
	X = dataset.iloc[:, :-1]
	y = dataset.iloc[:, -1:]

	return X, y

def cor_selector(dataset, threshold):
	corr = dataset.corr()
	columns = np.full((corr.shape[0], ), True, dtype = bool)
	for i in range(corr.shape[0]):
		for j in range(i + 1, corr.shape[0]):
			if corr.iloc[i,j] >= threshold:
				if columns[j]:
					columns[j] = False
	selected_columns = dataset.columns[columns]
	dataset = dataset[selected_columns]
	return dataset


def shuffle_split_data(X, y):
	arr_rand = np.random.rand(X.shape[0])
	split = arr_rand < np.percentile(arr_rand, 90)

	X_train = X[split]
	y_train = y[split]
	X_test =  X[~split]
	y_test = y[~split]

	return X_train, y_train, X_test, y_test

def nn_classification(name, frame, k_folds, learning_rate, epochs, layer_number):

	nb_features = len(frame.columns) - 1
	nb_classes  = frame["y"].nunique()
	start = datetime.now()

	# binary classification problem
	if name == "Cancer":
		total_accuracy = []

		for i in range(k_folds):
			X, y = split_xy(frame)
			X = stats.zscore(X)
			x_train, y_train, x_test, y_test = shuffle_split_data(X, y)

			nn = NeuralNetwork()
			
			input_layer  = Layer(x_train.shape[1], 10, 'tanh')
			hidden_layer = Layer(10, 10, 'sigmoid')
			output_layer = Layer(10, nb_classes, 'sigmoid')
			
			if layer_number == 2:
				nn.add_layer(input_layer)
				nn.add_layer(hidden_layer)
				nn.add_layer(output_layer)
			if layer_number == 1:
				nn.add_layer(input_layer)
				nn.add_layer(output_layer)
			if layer_number == 0:
				nn.add_layer(input_layer)

			errors = nn.train(x_train, y_train.values, learning_rate, epochs)
			accuracy = nn.accuracy(nn.predict(x_test)[:, 0].T.flatten(), y_test.values.flatten())
			total_accuracy.append(accuracy)

		print("Dataset: ", name)
		print("Hidden layers: ", layer_number)
		print("CV Scores: ", total_accuracy)
		print("Mean accuracy: ", sum(total_accuracy) / len(total_accuracy))
		print("Running time: ", datetime.now() - start)
		
		# added for testing
		if layer_number == 2:
			print("Input layer final weights: \t", input_layer.weights[0]) 
			print("Hidden layer final weights: \t", hidden_layer.weights[0])
			print("Output layer final weights: \t", output_layer.weights[0])
			print("Sample outputs from network: \t", errors[0:5])
		if layer_number == 1:
			print("Input layer final weights: \t", input_layer.weights[0]) 
			print("Output layer final weights: \t", output_layer.weights[0])
			print("Sample outputs from network: \t", errors[0:5])
		if layer_number == 0:
			print("Input layer final weights: \t", input_layer.weights[0]) 
			print("Sample outputs from network: \t", errors[0:5])
		nn.get_final_info()

	# multi-class classification
	else:
		total_accuracy = []

		for i in range(k_folds):
			attributes = frame.iloc[:, :nb_features]
			classes = frame.iloc[:, nb_features:]
			classes_encoded = pd.get_dummies(classes, columns = classes.columns)

			if name == "Glass":
				attributes = stats.zscore(attributes)
				x_train, y_train, x_test, y_test = shuffle_split_data(attributes, classes_encoded.values)
			else:
				x_train, y_train, x_test, y_test = shuffle_split_data(attributes, classes_encoded.values)

			nn = NeuralNetwork()

			input_layer  = Layer(x_train.shape[1], 10, 'tanh')
			hidden_layer = Layer(10, 10, 'sigmoid')
			output_layer = Layer(10, nb_classes, 'softmax')
			
			if layer_number == 2:
				nn.add_layer(input_layer)
				nn.add_layer(hidden_layer)
				nn.add_layer(output_layer)
			if layer_number == 1:
				nn.add_layer(input_layer)
				nn.add_layer(output_layer)
			if layer_number == 0:
				nn.add_layer(Layer(x_train.shape[1], nb_classes, 'softmax'))
			
			errors = nn.train(x_train, y_train, learning_rate, epochs)
			accuracy = nn.accuracy(nn.predict(x_test).T.flatten(), y_test.flatten())
			total_accuracy.append(accuracy)

		print("Dataset: ", name)
		print("Hidden layers: ", layer_number)
		print("CV Scores: ", total_accuracy)
		print("Mean accuracy: ", sum(total_accuracy) / len(total_accuracy))
		print("Running time: ", datetime.now() - start)

def nn_regression(name, frame, learning_rate, epochs):

	nb_features = len(frame.columns) - 1
	nb_classes  = frame["y"].nunique()
	start = datetime.now()

	X, y = split_xy(frame)
	X = stats.zscore(X)
	x_train, y_train, x_test, y_test = shuffle_split_data(X, y)

	nn = NeuralNetwork()
	nn.add_layer(Layer(x_train.shape[1], 5, 'sigmoid'))
	nn.add_layer(Layer(5, 3, 'sigmoid'))
	nn.add_layer(Layer(3, 1, 'sigmoid'))

	errors = nn.train(x_train, y_train.values, learning_rate, epochs)

	print("Dataset: ", name)
	print("MSE: ", nn.mse_metric(nn.predict(x_test)[:, 0].T.flatten(), y_test.values.flatten()))
	print("R2: ", nn.get_r2(nn.predict(x_test)[:, 0].T.flatten(), y_test.values.flatten()))
	print("Running time: ", datetime.now() - start)
