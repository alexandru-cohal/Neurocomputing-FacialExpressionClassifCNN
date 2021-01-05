# Author: Alexandru Cohal - 2018
# Based on: https://burakhimmetoglu.com/2017/08/22/time-series-classification-with-tensorflow/

import acquisition_fcn as acquisition
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import time

path = 'rawdata_01-03-18/'
person = 'v'
classes_used_list = ["neutral", "bb", "br", "bl", "om", "fm"]
n_classes = len(classes_used_list)
channels_used_list = range(14)
n_channels = len(channels_used_list)

def classifier_CNN(window_size, overlapping_size, artificial_size, batch, epoch, layers):

	no_artificial_add = 0

	# Read the recording for the specified classes. 
	# Obtain the values of the specified channels in a numpy array and the labels in another numpy array 
	inputs, labels = acquisition.read_recordings(path, person, classes_used_list, channels_used_list, window_size, overlapping_size, artificial_size)

	training_size = 0.6
	validating_size = 0.2
	testing_size = 0.2

	# Split the inputs and the labels into training, validation and testing datasets
	x_train, x_valid, x_test, lab_train, lab_valid, lab_test = acquisition.generate_train_valid_test_datasets(inputs, labels, training_size, validating_size, testing_size)

	x_train = acquisition.normalize(x_train)
	x_valid = acquisition.normalize(x_valid)
	x_test = acquisition.normalize(x_test)

	y_train = acquisition.one_hot(lab_train, n_classes)
	y_valid = acquisition.one_hot(lab_valid, n_classes)
	y_test = acquisition.one_hot(lab_test, n_classes)

	# Hyperparameters
	batch_size = batch
	seq_len = window_size
	learning_rate = 0.0001
	epochs = epoch

	# Construct placeholders
	graph = tf.Graph()

	with graph.as_default():
	    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
	    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
	    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
	    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

	# Build Convolutional Layers
	with graph.as_default():
		# (batch, 32, 14) --> (batch, 16, 28)
		conv1 = tf.layers.conv1d(inputs=inputs_, filters=28, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu)
		max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
		
		if layers > 1:
			# (batch, 16, 28) --> (batch, 8, 56)
			conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=56, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu)
			max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

		if layers > 2:
			# (batch, 8, 56) --> (batch, 4, 112)
			conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=112, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu)
			max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

		if layers > 3:
			# (batch, 4, 112) --> (batch, 2, 224)
			conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=224, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu)
			max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

	# Flatten and pass the classifier
	with graph.as_default():
		# Flatten and add dropout
		if layers == 1:
			flat = tf.reshape(max_pool_1, (-1, 16*28))
		elif layers == 2:
			flat = tf.reshape(max_pool_2, (-1, 8*56))
		elif layers == 3:
			flat = tf.reshape(max_pool_3, (-1, 4*112))
		elif layers == 4:
			flat = tf.reshape(max_pool_4, (-1, 2*224))

		flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

		# Predictions
		logits = tf.layers.dense(flat, n_classes)

		# Cost function and optimizer
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
		optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

		# Accuracy
		correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

	# Train the network
	if (os.path.exists('checkpoints-cnn') == False):
	    os.mkdir('checkpoints-cnn')

	print "Train!"
	start_time = time.time()

	validation_acc = []
	validation_loss = []

	train_acc = []
	train_loss = []

	with graph.as_default():
	    saver = tf.train.Saver()

	with tf.Session(graph=graph) as sess:
	    sess.run(tf.global_variables_initializer())
	    iteration = 1
	   
	    # Loop over epochs
	    for e in range(epochs):
		
		# Loop over batches
		for x,y in acquisition.get_batches(x_train, y_train, batch_size):
		    
		    # Feed dictionary
		    feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
		    
		    # Loss
		    loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
		    train_acc.append(acc)
		    train_loss.append(loss)
		    
		    # Print at each 5 iters
		    '''
		    if (iteration % 5 == 0):
		        print("Epoch: {}/{}".format(e, epochs),
		              "Iteration: {:d}".format(iteration),
		              "Train loss: {:6f}".format(loss),
		              "Train acc: {:.6f}".format(acc))
		    '''
		    
		    # Compute validation loss at every 10 iterations
		    if (iteration%5 == 0):                
		        val_acc_ = []
		        val_loss_ = []
		        
		        for x_v, y_v in acquisition.get_batches(x_valid, y_valid, batch_size):
		            # Feed
		            feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
		            
		            # Loss
		            loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
		            val_acc_.append(acc_v)
		            val_loss_.append(loss_v)
		        
		        # Print info
			'''
		        print("Epoch: {}/{}".format(e, epochs),
		              "Iteration: {:d}".format(iteration),
		              "Validation loss: {:6f}".format(np.mean(val_loss_)),
		              "Validation acc: {:.6f}".format(np.mean(val_acc_)))
			'''
		        
		        # Store
		        validation_acc.append(np.mean(val_acc_))
		        validation_loss.append(np.mean(val_loss_))
		    
		    # Iterate 
		    iteration += 1
	    
	    saver.save(sess,"checkpoints-cnn/har.ckpt")

	train_time = time.time() - start_time
	print "Done Train!"
	print "Training Time: ", train_time, " s"

	# Evaluate on testing dataset
	test_acc = []

	with tf.Session(graph=graph) as sess:
	    # Restore
	    saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))
	    
	    for x_t, y_t in acquisition.get_batches(x_test, y_test, batch_size):
		feed = {inputs_: x_t,
		        labels_: y_t,
		        keep_prob_: 1}
		
		batch_acc = sess.run(accuracy, feed_dict=feed)
		test_acc.append(batch_acc)
	    test_acc_value = np.mean(test_acc)
	    print("Test accuracy: {:.6f}".format(test_acc_value))

	# Plot
	# Plot training and test loss
	t = np.arange(iteration-1)

	fig = plt.figure()
	fig.set_size_inches(9, 6)
	fig.suptitle('Testing Accuracy: ' + str(int(test_acc_value * 10000) / 100.0) + ' %; Training time: ' + str(int(train_time * 100) / 100.0) + ' s')

	ax = plt.subplot(1, 2, 1)
	ax.plot(t, np.array(train_loss), 'r-', t[t % 5 == 0], np.array(validation_loss), 'b-')
	ax.set_title("Loss of Training and Validation")
	ax.set_xlabel("Iteration")
	ax.set_ylabel("Loss")
	ax.grid(True)
	ax.legend(['Training', 'Validation'], loc='upper right')

	# Plot Accuracies
	ax = plt.subplot(1, 2, 2)
	ax.plot(t, np.array(train_acc), 'r-', t[t % 5 == 0], validation_acc, 'b-')
	ax.set_title("Accuracy of Training and Validation")
	ax.set_xlabel("Iteration")
	ax.set_ylabel("Accuray")
	ax.legend(['Training', 'Validation'], loc='upper right')
	ax.grid(True)

	#plt.show()
	param_text = 'layers_' + str(layers) + '_window_' + str(window_size) + '_overlap_' + str(overlapping_size) + '_artif_' + str(artificial_size) + '_batch_' + str(batch_size) + '_epoch_' + str(epochs);
	plt.savefig('output/' + param_text + '.png')

	# Save the results
	results = [train_loss, validation_loss, train_acc, validation_acc, test_acc_value]
	np.save('output/' + param_text + '.npy', results)
