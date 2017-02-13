from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Convolution2D
from keras.layers import Reshape, SpatialDropout2D, Dropout, LeakyReLU
from keras.layers import MaxPooling2D, UpSampling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
from six.moves import cPickle as pickle
import time

# Select the class of images to use
# 0: Airplanes
# 1: Cars
# 2: Birds
# 3: Cats
# 4: Deer
# 5: Dogs
# 6: Frogs
# 7: Horses
# 8: Boats
# 9: Trucks

nb_epoch = 1000
display_interval = 1
save_interval = 100

disc_batch_size = 100
gen_batch_size = 32
gen_opt = Adam(lr=1e-4)
disc_opt = Adam(lr=1e-3)
gen_reg = l2(1e-5)
gen_loss_target = 1.1

for image_class in range(10):

	# Fetch data
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()

	# Verify data
	print X_train.shape, y_train.shape
	print X_test.shape, y_test.shape

	# Check distribution of classes
	y_dist = [0] * 10
	for y in y_train:
		y_dist[y[0]] += 1

	print y_dist

	# Convert uint8 pixel values to float32 in the range [0, 1] (for sigmoid)
	X_train = X_train.astype('float32')
	X_train /= 255

	# Split data into the target class and use the rest as unlabeled data
	X_new = np.ndarray(shape=(X_train.shape[0] / 10, X_train.shape[1], 
							  X_train.shape[2], X_train.shape[3]), dtype=np.float32)
	X_unl = np.ndarray(shape=(9 * X_train.shape[0] / 10, X_train.shape[1], 
							  X_train.shape[2], X_train.shape[3]), dtype=np.float32)
	sampleNumber = 0
	unlabeledNumber = 0
	for X, y in zip(X_train, y_train):
		if y[0] == image_class:
			X_new[sampleNumber] = X
			sampleNumber += 1
		else:
			X_unl[unlabeledNumber] = X
			unlabeledNumber += 1
	X_train = X_new[:, :, :, :]

	# Verify new data shapes
	print sampleNumber, unlabeledNumber
	print X_train.shape, X_unl.shape

	# Generator Model
	# Based on https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_cifar_feature_matching.py

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
	g_layer = Convolution2D(3, 5, 5, border_mode='same', W_regularizer=gen_reg)(g_layer)
	g_output = Activation('sigmoid')(g_layer)

	generator = Model(g_input, g_output)
	generator.compile(loss='binary_crossentropy', optimizer=gen_opt)
	# generator.summary()

	# Discriminator Model
	# Based on https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

	d_input = Input(shape=X_train.shape[1:])

	d_layer = Convolution2D(32, 3, 3, border_mode='same')(d_input)
	d_layer = Activation('relu')(d_layer)

	d_layer = Convolution2D(32, 3, 3)(d_layer)
	d_layer = Activation('relu')(d_layer)
	d_layer = MaxPooling2D(pool_size=(2, 2))(d_layer)
	d_layer = SpatialDropout2D(0.25)(d_layer)

	d_layer = Convolution2D(64, 3, 3, border_mode='same')(d_layer)
	d_layer = Activation('relu')(d_layer)

	d_layer = Convolution2D(64, 3, 3)(d_layer)
	d_layer = Activation('relu')(d_layer)
	d_layer = MaxPooling2D(pool_size=(2, 2))(d_layer)
	d_layer = SpatialDropout2D(0.25)(d_layer)

	d_layer = Flatten()(d_layer)
	d_layer = Dense(512)(d_layer)
	d_layer = Activation('relu')(d_layer)
	d_layer = Dropout(0.5)(d_layer)

	d_layer = Dense(1)(d_layer)
	d_output = Activation('sigmoid')(d_layer)

	discriminator = Model(d_input, d_output)
	discriminator.compile(loss='binary_crossentropy', optimizer=disc_opt)
	# discriminator.summary()

	# Helper functions

	def enable_training(model, setting = False):
		model.trainable = setting
		for layer in model.layers:
			layer.trainable = setting

	def accuracy(preds, ys):
		acc = 0
		for p, y in zip(preds, ys):
			if abs(p - y) < 0.5:
				acc += 1
		return 1. * acc / len(preds)

	def saveRecords(pickle_file):
		try:
			f = open(pickle_file, 'wb')
			pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
			f.close()
		except Exception as e:
			print 'Unable to save data to', pickle_file, ':', e
			raise

	# Disable training for discriminator layers
	enable_training(discriminator, False)
	
	# GAN Model
	gan_input = Input(shape=[100])
	gan_layer = generator(gan_input)
	gan_output = discriminator(gan_layer)

	GAN = Model(gan_input, gan_output)

	GAN.compile(loss='binary_crossentropy', optimizer=gen_opt)
	# GAN.summary()

	# Batch training functions

	def trainDiscriminator(batch_indices):
		noise = np.random.uniform(0, 1, size=[len(batch_indices) / 2, 100])
		generated_images = generator.predict(noise)
		unlabeled_images = X_unl[np.random.randint(0, X_unl.shape[0], size=len(batch_indices) / 2), :, :, :]
		
		X_batch = np.concatenate((X_train[batch_indices], generated_images, unlabeled_images))
		y_batch = np.zeros([2 * len(batch_indices)])
		y_batch[:len(batch_indices)] = 1 # Labels that are true images
		
		enable_training(discriminator, True) # Set layers to trainable
		loss  = discriminator.train_on_batch(X_batch, y_batch)
		enable_training(discriminator, False) # Set layers to not trainable
		
		return loss

	def trainGenerator(gen_batch_size):
		noise_batch = np.random.uniform(0, 1, size=[gen_batch_size, 100]) # New random inputs
		
		y_batch = np.ones([noise_batch.shape[0]]) # GAN attempts to make generated images true images
		
		loss = GAN.train_on_batch(noise_batch, y_batch)
		
		return loss

	# Pre-train discriminator
	noise = np.random.uniform(0, 1, size=[int(X_train.shape[0] / 2), 100])
	generated_images = generator.predict(noise)
	unlabeled_images = X_unl[np.random.randint(0, X_unl.shape[0], size=int(X_train.shape[0] / 2)), :, :, :]
	X_pre = np.concatenate((X_train, generated_images, unlabeled_images))
	y_pre = np.zeros([2 * X_train.shape[0]])
	y_pre[:X_train.shape[0]] = 1

	enable_training(discriminator, True)
	discriminator.fit(X_pre, y_pre, nb_epoch=1, batch_size=32, verbose=1)
	preds = discriminator.predict(X_pre)

	print "\nAccuracy:", accuracy(preds, y_pre)

	batches_per_epoch = X_train.shape[0] / disc_batch_size

	start = time.time()
	record = {"disc": [], "gen": [], "acc_real": [], "acc_gen": [], "acc_unl": []}
	for epoch in range(nb_epoch):

		# Create random batches of data to loop through all samples each epoch
		new_indices = range(X_train.shape[0])
		np.random.shuffle(new_indices)
		
		d_loss = 0
		g_loss = 0
		for batch_number in range(batches_per_epoch):
			batch_start = batch_number * disc_batch_size
			batch_end = (batch_number + 1) * disc_batch_size
			batch_indices = new_indices[batch_start:batch_end]

			d_loss += trainDiscriminator(batch_indices)

			# Always let generator reach competitive loss each batch
			g_batch_loss = float('inf')
			while g_batch_loss > gen_loss_target:
				g_batch_loss = trainGenerator(gen_batch_size)
			g_loss += g_batch_loss
		
		# Average loss across batches
		d_loss /= batches_per_epoch
		g_loss /= batches_per_epoch
		record["disc"].append(d_loss)
		record["gen"].append(g_loss)

		# Checking accuracy on a random subset of the data to save time
		# Check accuracy of discriminator on real images
		y_check = np.ones([X_train.shape[0] / 10])
		preds = discriminator.predict(
			X_train[np.random.randint(0, X_train.shape[0], size=X_train.shape[0] / 10), :, :, :])
		acc_real = accuracy(preds, y_check)
		record['acc_real'].append(acc_real)

		# Check accuracy of discriminator on generated images
		noise = np.random.uniform(0, 1, size=[X_train.shape[0] / 10, 100])
		X_check = generator.predict(noise)
		y_check = np.zeros([X_train.shape[0] / 10])
		preds = discriminator.predict(X_check)
		acc_gen = accuracy(preds, y_check)
		record['acc_gen'].append(acc_gen)

		# Check accuracy of discriminator on unlabeled images
		y_check = np.zeros([X_train.shape[0] / 10])
		preds = discriminator.predict(
			X_unl[np.random.randint(0, X_unl.shape[0], size=X_train.shape[0] / 10), :, :, :])
		acc_unl = accuracy(preds, y_check)
		record['acc_unl'].append(acc_unl)

		# Display progress:
		if epoch % display_interval == 0:
			print "Epoch: {:d}, d_loss: {:.2f}, g_loss: {:.2f}, real: {:.2f}, gen: {:.2f}, unl: {:.2f}".format(
								epoch, d_loss, g_loss, acc_real, acc_gen, acc_unl)
			print "Time elapsed:", (time.time() - start) / 60., "minutes"
			
		# Save progress to file
		if epoch % save_interval == 0 and epoch != 0:
			gen_file = "gen" + str(image_class) + "-" + str(epoch) + ".h5"
			generator.save(gen_file)
			print "Generator model saved to", gen_file
			pickle_file = "record" + str(image_class) + "-" + str(epoch) + ".pickle"
			saveRecords(pickle_file)
			print "Loss records saved to", pickle_file, "\n"
			
	print "Training time:", (time.time() - start) / 60., "minutes."

	# Save final results
	gen_file = "gen" + str(image_class) + "-" + str(nb_epoch) + ".h5"
	disc_file = "disc" + str(image_class) + "-" + str(nb_epoch) + ".h5"
	generator.save(gen_file)
	discriminator.save(disc_file)

	print "Final generator loss:", record["gen"][-1]
	print "Final discriminator loss:", record["disc"][-1]
	print "Recent accuracy on real images:", np.mean(record["acc_real"][-100:])
	print "Recent accuracy on generated images:", np.mean(record["acc_gen"][-100:])
	print "Recent accuracy on unlabeled images:", np.mean(record["acc_unl"][-100:])

	pickle_file = "record" + str(image_class) + "-" + str(nb_epoch) + ".pickle"
	saveRecords(pickle_file)