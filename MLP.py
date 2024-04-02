from LogisticAndNaiveBayes import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def _outputLayer_error(y, yhat):
	error = (yhat-y)*yhat*(1-yhat)
	return error

def _hiddenLayer_error(w2, output_error, h1):
    # Calculate the derivative of the activation function at h1
    derivative_h1 = h1 * (1 - h1)
    
    # Calculate the error for the hidden layer
    # This assumes that output_error is a scalar and w2 is a 1D array
    hidden_error = output_error * w2 * derivative_h1
    return hidden_error

def mlp(xTrain, xTest, yTrain, yTest, iNeurons: int = 2, oNeurons: int = 1, epochs: int = 1, threshold: float = 0.00001):
	temp = []
	for i in range(xTrain.shape[1]):
		temp.append([0.5]*iNeurons)
	w1 = np.array([]+temp)
	w2 = np.array([0.5]*iNeurons)
	b1 = np.array([0.5]*iNeurons)
	b2 = np.array([0.5]*oNeurons)
	initialWeights = [w1,w2]
	initialBiases = [b1,b2]
	alpha = 0.9
	delta_loss = 1
	last_loss = 1.0


	# Activation
	activation_function = lambda x: 1.0/(1.0 + np.exp(-x))

    # Training for a fixed number of epochs or until the loss is below a threshold
	for epoch in range(epochs):
		total_loss = 0
		for i in range(xTrain.shape[0]):
			# Forward pass
			h1 = activation_function(np.dot(xTrain[i], w1) + b1)
			yhat = activation_function(np.dot(h1, w2) + b2)

			# Calculate output layer error
			output_error = _outputLayer_error(yTrain[i], yhat)

			# Update output weights
			w2 = w2 - alpha * output_error * h1

			# Calculate hidden layer error
			hidden_error = _hiddenLayer_error(w2, output_error, h1)

			# Update hidden weights
			w1 = w1 - alpha * np.outer(xTrain[i], hidden_error)
			# # Update hidden weights (without using np.outer)
			# for j in range(iNeurons):
			# 	for k in range(xTrain.shape[1]):
			# 		w1[k, j] -= alpha * hidden_error[j] * xTrain[i, k]


			# Update biases?
			b1 = b1 - alpha * hidden_error
			b2 = b2 - alpha * output_error

			# Calculate loss (for monitoring purposes, not used in weight updates)
			loss = np.square(yTrain[i] - yhat).mean()
			total_loss += loss

		# Monitor the loss to decide when to stop
		total_loss /= xTrain.shape[0]
		last_delta_loss = delta_loss
		delta_loss = (last_loss - total_loss) ** 2
		last_loss = total_loss
		print(f"\rEpoch {epoch}, Delta Loss: {delta_loss}",end="")

		if delta_loss > last_delta_loss:
			print("Loss below threshold, stopping training.")
			break

	print()
	# After the training phase:
	# Test with the starting weights
	y_pred = []
	for i in range(xTest.shape[0]):
		# Forward pass with the test data
		h1_test = activation_function(np.dot(xTest[i], initialWeights[0]) + initialBiases[0])
		yhat_test = activation_function(np.dot(h1_test, initialWeights[1]) + initialBiases[1])
		# Apply a threshold to determine the class label
		y_pred.append(1 if yhat_test >= 0.5 else 0)

	# Calculate accuracy
	y_pred = np.array(y_pred)
	accuracy = np.mean(y_pred == yTest)
	print(f"Initial Weights Accuracy: {accuracy}")

	# Testing phase
	y_pred = []
	for i in range(xTest.shape[0]):
		# Forward pass with the test data
		h1_test = activation_function(np.dot(xTest[i], w1) + b1)
		yhat_test = activation_function(np.dot(h1_test, w2) + b2)
		# Apply a threshold to determine the class label
		y_pred.append(1 if yhat_test >= 0.5 else 0)

	# Calculate accuracy
	y_pred = np.array(y_pred)
	accuracy = np.mean(y_pred == yTest)
	print(f"Accuracy: {accuracy}")


def contains_not(text):
    return 1 if 'not' in text.split() else 0

def contains_security(text):
    return 1 if 'security' in text.split() else 0
    
if __name__ == "__main__":
	dataset = pd.read_csv('emails.csv', encoding='ISO-8859-1')
	feature_functions = [contains_not, contains_security]
	X, feature_names = bag_of_wordsify(dataset=dataset,feature_functions=feature_functions, max_token_features=50)

	y = dataset['spam']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	print(f"SHAPES: x {X_train.shape}, y {y_train.shape}")
	# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	y_train = y_train.reset_index(drop=True)
	y_test = y_test.reset_index(drop=True)
	mlp(X_train, X_test, y_train, y_test)

