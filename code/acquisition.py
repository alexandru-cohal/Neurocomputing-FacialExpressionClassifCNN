# Author: Alexandru Cohal - 2018
# Based on: https://burakhimmetoglu.com/2017/08/22/time-series-classification-with-tensorflow/

import os
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

LABEL_DICTIONARY = {"neutral":0, "bl":1, "br":2, "bb":3, "fm":4, "om":5, "eb":6, "m2l":7, "m2r":8, "n":9, "s":10}
NO_SAMPLES_IN_RECORDING = 100
NO_CHANNELS = 14
WINDOW_SIZE = 32
OVERLAPPING_SIZE = 16

def read_recordings(path, person, classes_used_list, channels_used_list, no_artificial_add):
	# Read all the recording files (each file contains a 110 by 14 matrix) for a specified person and for
	# specified classes and store the values for specified channels in a list ("inputs"). Create a list of 
	# corresponding labels ("labels") for each recording sample

	window_list = []
	label_list = []
	for file_name in sorted(os.listdir(path)):
		# Select only the recordings of the specified person
		if file_name[0] == person:
			# Get the class label (as string) of the current recording from the file name
			file_label = file_name.split("_")[1]
			# If it is one of the needed classes
			if file_label in classes_used_list:
				# Get the index of the class label (as int) using the class label dictionary
				index_label = LABEL_DICTIONARY[file_label]

				# Load the current recording and select the needed channels
				data_current = np.load(path + file_name)[:, channels_used_list]

				for window_start_index in range(0, NO_SAMPLES_IN_RECORDING, WINDOW_SIZE - OVERLAPPING_SIZE):
					if window_start_index + WINDOW_SIZE <= NO_SAMPLES_IN_RECORDING:
						window = data_current[window_start_index : window_start_index + WINDOW_SIZE, :]
						window_list.append(window)
						label_list.append(index_label)

						for artificial_add_index in range(no_artificial_add):
							noise = np.random.normal(20, 10, NO_CHANNELS)
							window_list.append(window + noise)
							label_list.append(index_label)

	segments = np.stack(window_list, axis=0)
	labels = np.asarray(label_list)

	return segments, labels


def generate_train_valid_test_datasets(inputs, labels, train_size, valid_size, test_size):
	x_train, x_valid_test, y_train, y_valid_test = train_test_split(inputs, labels, train_size = train_size)	
	x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, train_size = valid_size / (valid_size + test_size))

	return x_train, x_valid, x_test, y_train, y_valid, y_test
	

def get_batches(X, y, batch_size):
	# Return a single batch of X and y
	
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]

def one_hot(labels, n_class):
	# One-hot encoding used in machine learning
	
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"

	return y

def normalize(data):
	# Normalize the data - useful for machine learning

	data_norm = (data - np.mean(data, axis=0)[None,:,:]) / np.std(data, axis=0)[None,:,:]

	return data_norm