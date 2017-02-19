from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Flatten
from keras.layers import Input, Convolution2D, Reshape, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adadelta
import numpy as np
from six.moves import cPickle as pickle
import os
import time

# Helper functions
def accuracy(preds, ys):
	pred_class = np.argmax(preds, axis=1)
	true_class = np.argmax(ys, axis=1)
	difference = true_class - pred_class
	acc = (difference==0).sum()
	return 1. * acc / len(preds)

def saveRecords(pickle_file):
	try:
		f = open(pickle_file, 'wb')
		pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
		f.close()
	except Exception as e:
		print 'Unable to save data to', pickle_file, ':', e
		raise

for image_class in range(10):

	# Training parameters
	disc_optimizer = Adadelta(lr=1.0)
	disc_regularizer = l2(1e-4)

	disc_batch_size = 128
	nb_epoch = 300

	acc_check_size = 100
	display_interval = 1
	save_interval = 50

	# Fetch data
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()

	# Convert uint8 pixel values to float32 in the range [0, 1] (for sigmoid)
	X_train = X_train.astype('float32')
	X_train /= 255
	
	# Split data into the target class and use the rest as unlabeled data
	X_new = np.ndarray(shape=(X_train.shape[0] / 10, X_train.shape[1], 
						X_train.shape[2], X_train.shape[3]), dtype=np.float32)
	X_other = np.ndarray(shape=(9 * X_train.shape[0] / 10, X_train.shape[1], 
						X_train.shape[2], X_train.shape[3]), dtype=np.float32)
	sampleNumber = 0
	otherNumber = 0
	for X, y in zip(X_train, y_train):
		if y[0] == image_class:
			X_new[sampleNumber] = X
			sampleNumber += 1
		else:
			X_other[otherNumber] = X
			otherNumber += 1
	X_train = X_new[:, :, :, :]

	print sampleNumber, otherNumber
	print X_train.shape, X_other.shape

	# Discriminator Model
	# Based on https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

	d_input = Input(shape=X_train.shape[1:])

	d_layer = Convolution2D(32, 3, 3, border_mode='same', W_regularizer=disc_regularizer)(d_input)
	d_layer = LeakyReLU()(d_layer)

	d_layer = Convolution2D(32, 3, 3, border_mode='same')(d_layer)
	d_layer = LeakyReLU()(d_layer)
	d_layer = MaxPooling2D(pool_size=(2, 2))(d_layer) # 32 -> 16
	d_layer = SpatialDropout2D(0.25)(d_layer)

	d_layer = Convolution2D(64, 3, 3, border_mode='same', W_regularizer=disc_regularizer)(d_layer)
	d_layer = LeakyReLU()(d_layer)

	d_layer = Convolution2D(64, 3, 3, border_mode='same')(d_layer)
	d_layer = LeakyReLU()(d_layer)
	d_layer = MaxPooling2D(pool_size=(2, 2))(d_layer) # 16 -> 8
	d_layer = SpatialDropout2D(0.25)(d_layer)

	d_layer = Convolution2D(96, 3, 3, border_mode='same', W_regularizer=disc_regularizer)(d_layer)
	d_layer = LeakyReLU()(d_layer)

	d_layer = Convolution2D(96, 3, 3, border_mode='same')(d_layer)
	d_layer = LeakyReLU()(d_layer)
	d_layer = MaxPooling2D(pool_size=(2, 2))(d_layer) # 8 -> 4
	d_layer = SpatialDropout2D(0.25)(d_layer)

	d_layer = Convolution2D(128, 3, 3, border_mode='same', W_regularizer=disc_regularizer)(d_layer)
	d_layer = LeakyReLU()(d_layer)

	d_layer = Convolution2D(128, 3, 3, border_mode='same')(d_layer)
	d_layer = LeakyReLU()(d_layer)
	d_layer = MaxPooling2D(pool_size=(2, 2))(d_layer) # 4 -> 2
	d_layer = SpatialDropout2D(0.25)(d_layer)

	d_layer = Flatten()(d_layer) # 2 x 2 x 128 -> 512

	d_layer = Dense(256)(d_layer)
	d_layer = LeakyReLU()(d_layer)
	d_layer = Dropout(0.5)(d_layer)

	d_layer = Dense(256)(d_layer)
	d_layer = LeakyReLU()(d_layer)
	d_layer = Dropout(0.5)(d_layer)

	d_layer = Dense(2)(d_layer)
	d_output = Activation('softmax')(d_layer)

	discriminator = Model(d_input, d_output)
	discriminator.compile(loss='categorical_crossentropy', optimizer=disc_optimizer)
	# discriminator.summary()

	# Train discriminator
	record = {"disc": [], "gen": [], "acc_real": [], "acc_other": []}
	print "Starting training."
	
	start = time.time()
	batches_per_epoch = X_train.shape[0] / disc_batch_size
	for epoch in range(nb_epoch):

		d_loss = 0
		
		# Create random batches of data to loop through all samples each epoch
		new_indices = range(X_train.shape[0])
		np.random.shuffle(new_indices)

		for batch_number in range(batches_per_epoch):

			# Section of training data to use
			batch_start = batch_number * disc_batch_size
			batch_end = (batch_number + 1) * disc_batch_size

			# Make generative images
			image_batch = X_train[new_indices[batch_start:batch_end], :, :, :]    
			
			# Load random subset of non-class images
			other_images = X_other[np.random.randint(0, X_other.shape[0], size=int(disc_batch_size)), :, :, :]

			# Train discriminator
			X_batch = np.concatenate((image_batch, other_images))
			y_batch = np.zeros([2 * disc_batch_size, 2])
			y_batch[:disc_batch_size, 0] = 1 # True images
			y_batch[disc_batch_size:, 1] = 1 # Other images

			d_loss  += discriminator.train_on_batch(X_batch, y_batch)

		# Average loss across batches
		d_loss /= batches_per_epoch
		
		# Checking accuracy on a random subset of the data to save time
		# Check accuracy of discriminator on real images
		y_check = np.zeros([acc_check_size, 2])
		y_check[:, 0] = 1
		preds = discriminator.predict(X_train[np.random.randint(0, X_train.shape[0], 
																size=acc_check_size), :, :, :])
		acc_real = accuracy(preds, y_check)

		# Check accuracy of discriminator on other images
		y_check = np.zeros([acc_check_size, 2])
		y_check[:, 1] = 1
		preds = discriminator.predict(X_other[np.random.randint(0, X_other.shape[0], 
															  size=acc_check_size), :, :, :])
		acc_other = accuracy(preds, y_check)

		# Log progress
		record["disc"].append(d_loss)
		record['acc_real'].append(acc_real)
		record['acc_other'].append(acc_other)
		
		# Display progress:
		if epoch % display_interval == 0:
			msg = ""
			msg += "Epoch: {:d}, d_loss: {:.2f}".format(epoch, d_loss)
			msg += ", real: {:.2f}, other: {:.2f}".format(acc_real, acc_other)
			print msg
			msg = ""
			msg += "Time elapsed: {:.2f} minutes".format((time.time() - start) / 60)
			print msg

		# Save progress to file
		if (epoch + 1) % save_interval == 0 and (epoch + 1) != nb_epoch:
			disc_file = "disc" + str(image_class) + "-" + str(epoch + 1) + ".h5"
			discriminator.save(disc_file)
			print "\nDiscriminator model saved to", disc_file
			pickle_file = "record" + str(image_class) + "-" + str(epoch + 1) + ".pickle"
			saveRecords(pickle_file)
			print "Loss records saved to", pickle_file, "\n"

	print "Training completed in", (time.time() - start) / 60., "minutes."

	print "Discriminator loss:", record["disc"][-1]
	print "Accuracy on real images:", np.mean(record["acc_real"][-int(0.1 * nb_epoch):])
	print "Accuracy on other images:", np.mean(record["acc_other"][-int(0.1 * nb_epoch):])

	disc_file = "disc" + str(image_class) + "-" + str(nb_epoch) + ".h5"
	discriminator.save(disc_file)

	pickle_file = "record" + str(image_class) + "-" + str(nb_epoch) + ".pickle"
	saveRecords(pickle_file)
	statinfo = os.stat(pickle_file)
	print 'Compressed pickle size:', statinfo.st_size