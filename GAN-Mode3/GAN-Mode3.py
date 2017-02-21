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
def enable_training(model, setting = False):
	model.trainable = setting
	for layer in model.layers:
		layer.trainable = setting

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
	disc_optimizer = Adadelta()
	GAN_optimizer = Adadelta()

	gen_regularizer = l2(1e-4)
	disc_regularizer = l2(1e-4)

	disc_batch_size = 100
	gen_batch_size = 100
	g_loss_target = 1.2
	gen_batch_max = 10
	gen_batch_base = 2
	nb_epoch = 500

	acc_check_size = 100
	display_interval = 1
	save_interval = 10

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

	# The following models and training process were heavily influenced by the following three sources:
	# https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN_v2.ipynb
	# https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_cifar_minibatch_discrimination.py
	# http://torch.ch/blog/2015/11/13/gan.html

	# Generator Model
	g_input = Input(shape=[100])
	g_layer = Dense(4*4*512)(g_input)
	g_layer = BatchNormalization()(g_layer)
	g_layer = Activation('relu')(g_layer)

	g_layer = Reshape( [4, 4, 512] )(g_layer)

	g_layer = UpSampling2D(size=(2, 2))(g_layer) # 4 -> 8
	g_layer = Convolution2D(256, 5, 5, border_mode='same')(g_layer)
	g_layer = BatchNormalization(mode=2)(g_layer)
	g_layer = Activation('relu')(g_layer)

	g_layer = UpSampling2D(size=(2, 2))(g_layer) # 8 -> 16
	g_layer = Convolution2D(128, 5, 5, border_mode='same')(g_layer)
	g_layer = BatchNormalization(mode=2)(g_layer)
	g_layer = Activation('relu')(g_layer)

	g_layer = UpSampling2D(size=(2, 2))(g_layer) # 16 -> 32
	g_layer = Convolution2D(3, 5, 5, border_mode='same', W_regularizer=gen_regularizer)(g_layer)
	g_output = Activation('sigmoid')(g_layer)

	generator = Model(g_input, g_output)
	generator.compile(loss='mean_squared_error', optimizer=GAN_optimizer)
	# generator.summary()

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

	d_layer = Dense(3)(d_layer)
	d_output = Activation('softmax')(d_layer)

	discriminator = Model(d_input, d_output)
	discriminator.compile(loss='categorical_crossentropy', optimizer=disc_optimizer)
	# discriminator.summary()

	# Lock weights in the discriminator for training generator
	enable_training(discriminator, False)

	# GAN Model
	gan_input = Input(shape=[100])
	gan_layer = generator(gan_input)
	gan_output = discriminator(gan_layer)
	GAN = Model(gan_input, gan_output)
	GAN.compile(loss='categorical_crossentropy', optimizer=GAN_optimizer)
	# GAN.summary()

	# Pre-train discriminator
	noise = np.random.uniform(0, 1, size=[int(X_train.shape[0]), 100])
	generated_images = generator.predict(noise)
	other_images = X_other[np.random.randint(0, X_other.shape[0], size=int(X_train.shape[0])), :, :, :]
	X_pre = np.concatenate((X_train, generated_images, other_images))
	y_pre = np.zeros([3 * X_train.shape[0], 3])
	y_pre[:X_train.shape[0], 0] = 1 # True images
	y_pre[X_train.shape[0]:2 * X_train.shape[0], 1] = 1 # Generated images
	y_pre[2 * X_train.shape[0]:3 * X_train.shape[0], 2] = 1 # Other images

	enable_training(discriminator, True)
	discriminator.fit(X_pre, y_pre, nb_epoch=3, batch_size=32, verbose=2)
	preds = discriminator.predict(X_pre)

	print "Convergence:", accuracy(preds, y_pre)

	record = {"disc": [], "gen": [], "acc_real": [], "acc_gen": [], "acc_other": [], "gen_batches": []}
	print "Starting training."
	
	start = time.time()
	batches_per_epoch = X_train.shape[0] / disc_batch_size
	for epoch in range(nb_epoch):

		d_loss = 0
		g_loss = 0
		g_batch = 0

		# Create random batches of data to loop through all samples each epoch
		new_indices = range(X_train.shape[0])
		np.random.shuffle(new_indices)

		for batch_number in range(batches_per_epoch):

			# Section of training data to use
			batch_start = batch_number * disc_batch_size
			batch_end = (batch_number + 1) * disc_batch_size

			# Make generative images
			image_batch = X_train[new_indices[batch_start:batch_end], :, :, :]    
			noise_batch = np.random.uniform(0, 1, size=[int(disc_batch_size), 100])
			generated_images = generator.predict(noise_batch)
			
			# Load random subset of non-class images
			other_images = X_other[np.random.randint(0, X_other.shape[0], size=int(disc_batch_size)), :, :, :]

			# Train discriminator
			X_batch = np.concatenate((image_batch, generated_images, other_images))
			y_batch = np.zeros([3 * disc_batch_size, 3])
			y_batch[:disc_batch_size, 0] = 1 # True images
			y_batch[disc_batch_size:2 * disc_batch_size, 1] = 1 # Generated images
			y_batch[2 * disc_batch_size:3 * disc_batch_size, 2] = 1 # Other images

			enable_training(discriminator, True) # Set layers to trainable
			d_loss  += discriminator.train_on_batch(X_batch, y_batch)
			enable_training(discriminator, False) # Set layers to not trainable

			# Deliberately use a single noise vector in the GAN training loop below
			noise_batch = np.random.uniform(0, 1, size=[gen_batch_size, 100]) # New random inputs
			y_batch_2 = np.zeros([gen_batch_size, 3])
			y_batch_2[:, 0] = 1 # GAN attempts to make generated images true images
			
			# Train GAN
			# Ramp up the allowed number of training batches
			allowed_batches = int(gen_batch_base + min(epoch * (gen_batch_max - gen_batch_base) / 
											(0.5 * nb_epoch), gen_batch_max - gen_batch_base))
			
			g_batch_count = 0
			g_batch_loss = float('inf')
			while g_batch_loss > g_loss_target:
				g_batch_count += 1
				g_batch_loss = GAN.train_on_batch(noise_batch, y_batch_2)
				
				if g_batch_count >= allowed_batches:
					break
			
			g_loss += g_batch_loss
			g_batch += g_batch_count

		# Average loss across batches
		d_loss /= batches_per_epoch
		g_loss /= batches_per_epoch
		g_batch /= 1. * batches_per_epoch

		# Checking accuracy on a random subset of the data to save time
		# Check accuracy of discriminator on real images
		y_check = np.zeros([acc_check_size, 3])
		y_check[:, 0] = 1
		preds = discriminator.predict(X_train[np.random.randint(0, X_train.shape[0], 
																size=acc_check_size), :, :, :])
		acc_real = accuracy(preds, y_check)

		# Check accuracy of discriminator on generated images
		noise = np.random.uniform(0, 1, size=[acc_check_size, 100])
		X_check = generator.predict(noise)
		y_check = np.zeros([acc_check_size, 3])
		y_check[:, 1] = 1
		preds = discriminator.predict(X_check)
		acc_gen = accuracy(preds, y_check)

		# Check accuracy of discriminator on other images
		y_check = np.zeros([acc_check_size, 3])
		y_check[:, 2] = 1
		preds = discriminator.predict(X_other[np.random.randint(0, X_other.shape[0], 
															  size=acc_check_size), :, :, :])
		acc_other = accuracy(preds, y_check)

		# Log progress
		record["disc"].append(d_loss)
		record["gen"].append(g_loss)
		record['acc_real'].append(acc_real)
		record['acc_gen'].append(acc_gen)
		record['acc_other'].append(acc_other)
		record['gen_batches'].append(g_batch)
		
		# Display progress:
		if epoch % display_interval == 0:
			msg = ""
			msg += "Epoch: {:d}, d_loss: {:.2f}, g_loss: {:.2f}".format(epoch, d_loss, g_loss)
			msg += ", real: {:.2f}, gen: {:.2f}, other: {:.2f}".format(acc_real, acc_gen, acc_other)
			print msg
			msg = ""
			msg += "Time elapsed: {:.2f} minutes".format((time.time() - start) / 60)
			msg += ", generator batch count: {:.2f} of {:.2f}".format(g_batch, allowed_batches)
			print msg

		# Save progress to file
		if (epoch + 1) % save_interval == 0 and (epoch + 1) != nb_epoch:
			gen_file = "networks/gen" + str(image_class) + "-" + str(epoch + 1) + ".h5"
			generator.save(gen_file)
			print "\nGenerator model saved to", gen_file
			disc_file = "networks/disc" + str(image_class) + "-" + str(epoch + 1) + ".h5"
			discriminator.save(disc_file)
			print "Discriminator model saved to", disc_file
			pickle_file = "records/record" + str(image_class) + "-" + str(epoch + 1) + ".pickle"
			saveRecords(pickle_file)
			print "Loss records saved to", pickle_file, "\n"

	print "Training completed in", (time.time() - start) / 60., "minutes."

	print "Generator loss:", record["gen"][-1]
	print "Discriminator loss:", record["disc"][-1]
	print "Mean accuracy on real images:", np.mean(record["acc_real"][-int(0.1 * nb_epoch):])
	print "Mean accuracy on generated images:", np.mean(record["acc_gen"][-int(0.1 * nb_epoch):])
	print "Mean accuracy on other images:", np.mean(record["acc_other"][-int(0.1 * nb_epoch):])

	gen_file = "networks/gen" + str(image_class) + "-" + str(nb_epoch) + ".h5"
	disc_file = "networks/disc" + str(image_class) + "-" + str(nb_epoch) + ".h5"
	generator.save(gen_file)
	discriminator.save(disc_file)

	pickle_file = "records/record" + str(image_class) + "-" + str(nb_epoch) + ".pickle"
	saveRecords(pickle_file)
	statinfo = os.stat(pickle_file)
	print 'Compressed pickle size:', statinfo.st_size