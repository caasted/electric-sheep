from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, SpatialDropout2D, Activation, Flatten
from keras.layers import Input, Convolution2D, Reshape, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adagrad
from keras.utils.np_utils import to_categorical
# import keras.backend as K
import numpy as np
from six.moves import cPickle as pickle
import os
import time

# Training parameters
nb_epoch = 500

initial_lr = 1e-3
disc_optimizer = Adagrad(lr = initial_lr)
GAN_optimizer = Adagrad(lr = initial_lr)

gen_regularizer = l2(1e-4)
disc_regularizer = l2(1e-4)

disc_batch_size = 100
gen_batch_size = 100
noise_length = 100

acc_check_size = 100
display_interval = 1
save_interval = 50

# Fetch data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert uint8 pixel values to float32 in the range [-1, 1] (for tanh)
X_train = X_train.astype('float32')
X_train -= 128
X_train /= 128
X_test = X_test.astype('float32')
X_test -= 128
X_test /= 128

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Verify data
print X_train.shape, "Mean:", np.mean(X_train), "StdDev:", np.std(X_train)
print y_train.shape, "Class dist.:", np.sum(y_train, axis=0)

# The following models and training process were heavily influenced by the following sources:
# https://keras.io/getting-started/sequential-model-guide/#examples
# https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_cifar_minibatch_discrimination.py
# http://torch.ch/blog/2015/11/13/gan.html
# https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN_v2.ipynb

# Generator Model
g_input = Input(shape=[noise_length])
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
g_output = Activation('tanh')(g_layer)

generator = Model(g_input, g_output)
generator.compile(loss='mean_squared_error', optimizer=GAN_optimizer)
# generator.summary()

# Discriminator Model
d_input = Input(shape=X_train.shape[1:])

d_layer = Convolution2D(96, 3, 3, border_mode='same', W_regularizer=disc_regularizer)(d_input)
d_layer = LeakyReLU()(d_layer)

d_layer = Convolution2D(96, 3, 3, border_mode='same', W_regularizer=disc_regularizer)(d_input)
d_layer = LeakyReLU()(d_layer)

d_layer = Convolution2D(96, 3, 3, subsample=(2, 2), border_mode='same', 
						W_regularizer=disc_regularizer)(d_layer) # 32 -> 16
d_layer = LeakyReLU()(d_layer)
d_layer = SpatialDropout2D(0.5)(d_layer)

d_layer = Convolution2D(192, 3, 3, border_mode='same', W_regularizer=disc_regularizer)(d_layer)
d_layer = LeakyReLU()(d_layer)

d_layer = Convolution2D(192, 3, 3, border_mode='same', W_regularizer=disc_regularizer)(d_layer)
d_layer = LeakyReLU()(d_layer)

d_layer = Convolution2D(192, 3, 3, subsample=(2, 2), border_mode='same', 
						W_regularizer=disc_regularizer)(d_layer) # 16 -> 8
d_layer = LeakyReLU()(d_layer)
d_layer = SpatialDropout2D(0.5)(d_layer)

d_layer = Convolution2D(192, 1, 1, border_mode='same', W_regularizer=disc_regularizer)(d_layer)
d_layer = LeakyReLU()(d_layer)

d_layer = Convolution2D(192, 1, 1, border_mode='same', W_regularizer=disc_regularizer)(d_layer)
d_layer = LeakyReLU()(d_layer)

d_layer = Flatten()(d_layer)

d_output = Dense(20, activation='softmax', W_regularizer=disc_regularizer)(d_layer)

discriminator = Model(d_input, d_output)
discriminator.compile(loss='categorical_crossentropy', optimizer=disc_optimizer)
# discriminator.summary()

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

# Lock weights in the discriminator for training generator
enable_training(discriminator, False)

# GAN Model
gan_input = Input(shape=[noise_length])
gan_layer = generator(gan_input)
gan_output = discriminator(gan_layer)
GAN = Model(gan_input, gan_output)
GAN.compile(loss='categorical_crossentropy', optimizer=GAN_optimizer)
# GAN.summary()

# Pre-train discriminator
noise = np.random.uniform(0, 1, size=[X_train.shape[0], noise_length])
gen_class = np.random.randint(0, y_train.shape[1], size=[X_train.shape[0]])
for index in range(len(gen_class)):
	# Set 10 indices to zeros corresponding to gen_class
	noise[index, 10 * gen_class[index]:10 * (gen_class[index] + 1)] = 0

generated_images = generator.predict(noise)
X_pre = np.concatenate((X_train, generated_images))
y_pre = np.zeros((2 * y_train.shape[0], 2 * y_train.shape[1]))
y_pre[:y_train.shape[0], :y_train.shape[1]] = y_train
y_pre[y_train.shape[0]:, y_train.shape[1]:] = to_categorical(gen_class, y_train.shape[1])
print X_pre.shape, y_pre.shape

enable_training(discriminator, True)
discriminator.fit(X_pre, y_pre, nb_epoch=1, batch_size=disc_batch_size, verbose=2)
preds = discriminator.predict(X_pre)
acc = accuracy(preds, y_pre)
print "Accuracy: {:.4f}".format(acc)

def trainGAN():

	start = time.time()
	batches_per_epoch = X_train.shape[0] / disc_batch_size
	for epoch in range(nb_epoch):

		# Apply cosine learning rate decay with final value = 10% initial lr (tuned for 1200 nb_epoch)
		# lr_update = initial_lr * np.cos(epoch / (1.067222 * nb_epoch) * np.pi / 2)
		# K.set_value(discriminator.optimizer.lr, K.cast_to_floatx(lr_update))
		# K.set_value(generator.optimizer.lr, K.cast_to_floatx(lr_update))
		# K.set_value(GAN.optimizer.lr, K.cast_to_floatx(lr_update))

		d_loss = 0
		g_loss = 0

		# Create random batches of data to loop through all samples each epoch
		new_indices = range(X_train.shape[0])
		np.random.shuffle(new_indices)

		for batch_number in range(batches_per_epoch):

			# Section of training data to use
			batch_start = batch_number * disc_batch_size
			batch_end = (batch_number + 1) * disc_batch_size

			# Make generative images
			image_batch = X_train[new_indices[batch_start:batch_end], :, :, :]    
			noise_batch = np.random.uniform(0, 1, size=[disc_batch_size, noise_length])
			gen_class = np.random.randint(0, y_train.shape[1], size=[disc_batch_size])
			for index in range(len(gen_class)):
				# Set 10 indices to zeros corresponding to gen_class
				noise_batch[index, 10 * gen_class[index]:10 * (gen_class[index] + 1)] = 0

			generated_images = generator.predict(noise_batch)

			# Train discriminator
			X_batch = np.concatenate((image_batch, generated_images))

			y_batch = np.zeros((int(2 * disc_batch_size), 2 * y_train.shape[1]))
			y_batch[:disc_batch_size, :y_train.shape[1]] = y_train[new_indices[batch_start:batch_end]]
			y_batch[disc_batch_size:, y_train.shape[1]:] = to_categorical(gen_class, y_train.shape[1])

			enable_training(discriminator, True) # Set layers to trainable
			d_loss  += discriminator.train_on_batch(X_batch, y_batch)
			enable_training(discriminator, False) # Set layers to not trainable

			noise_batch = np.random.uniform(0, 1, size=[gen_batch_size, noise_length])
			gen_class = np.random.randint(0, y_train.shape[1], size=[gen_batch_size])
			for index in range(len(gen_class)):
				# Set 10 indices to zeros corresponding to gen_class
				noise_batch[index, 10 * gen_class[index]:10 * (gen_class[index] + 1)] = 0

			y_batch2 = np.zeros((gen_batch_size, 2 * y_train.shape[1]))
			# Set generated class to train as a random true class
			y_batch2[:, :y_train.shape[1]] = to_categorical(gen_class, y_train.shape[1])

			# Train GAN
			g_loss += GAN.train_on_batch(noise_batch, y_batch2)

		# Average loss across batches
		d_loss /= batches_per_epoch
		g_loss /= batches_per_epoch
		
		# Checking accuracy on a random subset of the data to save time
		# Check accuracy of discriminator on real images
		check_indices = np.random.randint(0, X_train.shape[0], size=acc_check_size)
		y_check = np.zeros((acc_check_size, 2 * y_train.shape[1]))
		y_check[:, :y_train.shape[1]] = y_train[check_indices, :]
		preds = discriminator.predict(X_train[check_indices, :, :, :])
		acc_real = accuracy(preds, y_check)

		# Check accuracy of discriminator on generated images
		noise = np.random.uniform(0, 1, size=[acc_check_size, 100])
		gen_class = np.random.randint(0, y_train.shape[1], size=[acc_check_size])
		for index in range(len(gen_class)):
			# Set 10 indices to zeros corresponding to gen_class
			noise[index, 10 * gen_class[index]:10 * (gen_class[index] + 1)] = 0

		X_check = generator.predict(noise)
		y_check = np.zeros((acc_check_size, 2 * y_train.shape[1]))
		y_check[:, y_train.shape[1]:] = to_categorical(gen_class, y_train.shape[1])
		preds = discriminator.predict(X_check)
		acc_gen = accuracy(preds, y_check)

		
		### SUPER TEMPORARY ###

		# Check accuracy of discriminator on actual test images
		preds = discriminator.predict(X_test)
		preds = preds % y_test.shape[1] # Fold generated and real predictions together
		y_secret = np.zeros((y_test.shape[0], 2 * y_test.shape[1]))
		y_secret[:, :y_test.shape[1]] = y_test
		acc_test = accuracy(preds, y_secret)
		record['acc_test'].append(acc_test)

		### DELETE AFTER CHECKING ###


		# Log progress
		record["disc"].append(d_loss)
		record["gen"].append(g_loss)
		record['acc_real'].append(acc_real)
		record['acc_gen'].append(acc_gen)
		
		# Display progress:
		if epoch % display_interval == 0:
			msg = ""
			msg += "Epoch: {:d}, d_loss: {:.2f}, g_loss: {:.2f}".format(epoch, d_loss, g_loss)
			msg += ", real: {:.2f}, gen: {:.2f}, test: {:.4f}".format(acc_real, acc_gen, acc_test)
			print msg
			msg = ""
			msg += "Time elapsed: {:.2f} minutes".format((time.time() - start) / 60)
			print msg

		# Save progress to file
		if (epoch + 1) % save_interval == 0 and (epoch + 1) != nb_epoch:
			gen_file = "generator-" + str(epoch + 1) + ".h5"
			disc_file = "discriminator-" + str(epoch + 1) + ".h5"
			generator.save(gen_file)
			discriminator.save(disc_file)
			print "\nGenerator model saved to", gen_file
			print "\nDiscriminator model saved to", disc_file
			pickle_file = "record-" + str(epoch + 1) + ".pickle"
			saveRecords(pickle_file)
			print "Loss records saved to", pickle_file, "\n"

	print "Training completed in", (time.time() - start) / 60., "minutes."
	return epoch + 1

record = {"disc": [], "gen": [], "acc_real": [], "acc_gen": [], "acc_test": []}

print "Starting training."
final_epoch = trainGAN()
print "Training complete."

print "Generator loss:", record["gen"][-1]
print "Discriminator loss:", record["disc"][-1]
print "Accuracy on real images:", np.mean(record["acc_real"][-100:])
print "Accuracy on generated images:", np.mean(record["acc_gen"][-100:])

gen_file = "generator-" + str(final_epoch) + ".h5"
disc_file = "discriminator-" + str(final_epoch) + ".h5"
generator.save(gen_file)
discriminator.save(disc_file)

pickle_file = "record-" + str(final_epoch) + ".pickle"
saveRecords(pickle_file)
statinfo = os.stat(pickle_file)
print 'Compressed pickle size:', statinfo.st_size