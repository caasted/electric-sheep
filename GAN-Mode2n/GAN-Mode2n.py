from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Flatten
from keras.layers import Input, Convolution2D, Reshape, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, Adadelta
from keras.utils.np_utils import to_categorical
import keras.backend as K
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

# Training parameters
# learning_rate = 1e-4
# disc_optimizer = Adam(lr=learning_rate)
# GAN_optimizer = Adam(lr=learning_rate)
disc_optimizer = Adadelta()
GAN_optimizer = Adadelta()

gen_regularizer = l2(1e-4)
disc_regularizer = l2(1e-4)

disc_batch_size = 100 # Has to be a multiple of 10
gen_batch_size = 100  # Has to be a multiple of 10
g_loss_target = 1.2
gen_batch_max = 10
gen_batch_base = 2
nb_epoch = 1000

input_noise_size = 100 # Size of noise array for each category of generated image

acc_check_size = 100 # Has to be a multiple of 10
display_interval = 1
save_interval = 10

# Fetch data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert uint8 pixel values to float32 in the range [0, 1] (for sigmoid)
X_train = X_train.astype('float32')
X_train /= 255

# Convert y_train data to categorical format
y_train = to_categorical(y_train)

# The following models and training process were heavily influenced by the following sources:
# https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_cifar_minibatch_discrimination.py
# https://arxiv.org/abs/1411.1784
# http://torch.ch/blog/2015/11/13/gan.html
# https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN_v2.ipynb

# Generator Model
g_input = Input(shape=[input_noise_size + y_train.shape[1]])
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

d_layer = Dense(2 * y_train.shape[1])(d_layer)
d_output = Activation('softmax')(d_layer)

discriminator = Model(d_input, d_output)
discriminator.compile(loss='categorical_crossentropy', optimizer=disc_optimizer)
# discriminator.summary()

# Lock weights in the discriminator for training generator
enable_training(discriminator, False)

# GAN Model
gan_input = Input(shape=[input_noise_size + y_train.shape[1]])
gan_layer = generator(gan_input)
gan_output = discriminator(gan_layer)
GAN = Model(gan_input, gan_output)
GAN.compile(loss='categorical_crossentropy', optimizer=GAN_optimizer)
# GAN.summary()

# Pre-train discriminator
noise = np.random.uniform(0, 1, size=[y_train.shape[0], input_noise_size])
noise_class = np.zeros([y_train.shape[0], y_train.shape[1]])
for image_class in range(y_train.shape[1]):
	start_row = image_class * y_train.shape[0] / y_train.shape[1]
	end_row = (image_class + 1) * y_train.shape[0] / y_train.shape[1]
	noise_class[start_row:end_row, image_class] = 1
noise = np.concatenate((noise, noise_class), axis=1)

generated_images = generator.predict(noise)
X_pre = np.concatenate((X_train, generated_images))

y_pre = np.zeros([2 * y_train.shape[0], 2 * y_train.shape[1]])
y_pre[:y_train.shape[0], :y_train.shape[1]] = y_train # True images
for image_class in range(y_train.shape[1]):
	start_row = y_train.shape[0] + image_class * y_train.shape[0] / y_train.shape[1]
	end_row = y_train.shape[0] + (image_class + 1) * y_train.shape[0] / y_train.shape[1]
	y_pre[start_row:end_row, y_train.shape[1] + image_class] = 1 # Generated images

enable_training(discriminator, True)
discriminator.fit(X_pre, y_pre, nb_epoch=3, batch_size=32, verbose=2)
preds = discriminator.predict(X_pre)

print "Convergence:", accuracy(preds, y_pre)

record = {"disc": [], "gen": [], "acc_real": [], "acc_gen": [], "gen_batches": []}
print "Starting training."

start = time.time()
batches_per_epoch = X_train.shape[0] / disc_batch_size
for epoch in range(nb_epoch):

	# For use with Adam optimizer, not Adadelta
	# Apply cosine learning rate decay with final value = 10% initial lr (tuned for 1000 nb_epoch)
	# lr_update = learning_rate * np.cos(epoch / ( 1.067044 * nb_epoch) * np.pi / 2)
	# K.set_value(discriminator.optimizer.lr, K.cast_to_floatx(lr_update))
	# K.set_value(GAN.optimizer.lr, K.cast_to_floatx(lr_update))
	# K.set_value(generator.optimizer.lr, K.cast_to_floatx(lr_update)) # Just to be certain

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
		image_batch = X_train[new_indices[batch_start:batch_end]]    
		
		noise = np.random.uniform(0, 1, size=[disc_batch_size, input_noise_size])
		noise_class = np.zeros([disc_batch_size, y_train.shape[1]])
		for image_class in range(y_train.shape[1]):
			start_row = image_class * disc_batch_size / y_train.shape[1]
			end_row = (image_class + 1) * disc_batch_size / y_train.shape[1]
			noise_class[start_row:end_row, image_class] = 1
		noise_batch = np.concatenate((noise, noise_class), axis=1)

		generated_images = generator.predict(noise_batch)
		
		# Train discriminator
		X_batch = np.concatenate((image_batch, generated_images))
		y_batch = np.zeros([2 * disc_batch_size, 2 * y_train.shape[1]])
		y_batch[:disc_batch_size, :y_train.shape[1]] = y_train[new_indices[batch_start:batch_end]] # True images
		for image_class in range(y_train.shape[1]):
			start_row = disc_batch_size + image_class * disc_batch_size / y_train.shape[1]
			end_row = disc_batch_size + (image_class + 1) * disc_batch_size / y_train.shape[1]
			y_batch[start_row:end_row, y_train.shape[1] + image_class] = 1 # Generated images

		enable_training(discriminator, True) # Set layers to trainable
		d_loss  += discriminator.train_on_batch(X_batch, y_batch)
		enable_training(discriminator, False) # Set layers to not trainable

		# Deliberately use a single noise vector in the GAN training loop below
		noise = np.random.uniform(0, 1, size=[gen_batch_size, input_noise_size])
		noise_class = np.zeros([gen_batch_size, y_train.shape[1]])
		for image_class in range(y_train.shape[1]):
			start_row = image_class * gen_batch_size / y_train.shape[1]
			end_row = (image_class + 1) * gen_batch_size / y_train.shape[1]
			noise_class[start_row:end_row, image_class] = 1
		noise_batch = np.concatenate((noise, noise_class), axis=1)
		
		y_batch = np.zeros([gen_batch_size, 2 * y_train.shape[1]])
		for image_class in range(y_train.shape[1]):
			start_row = image_class * gen_batch_size / y_train.shape[1]
			end_row = (image_class + 1) * gen_batch_size / y_train.shape[1]
			y_batch[start_row:end_row, image_class] = 1 # Backprop generated images to resemble true images
		
		# Train GAN
		# Ramp up the allowed number of training batches
		allowed_batches = int(gen_batch_base + min(epoch * (gen_batch_max - gen_batch_base) / 
										(0.5 * nb_epoch), gen_batch_max - gen_batch_base))
		
		g_batch_count = 0
		g_batch_loss = float('inf')
		while g_batch_loss > g_loss_target:
			g_batch_count += 1
			g_batch_loss = GAN.train_on_batch(noise_batch, y_batch)
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
	check_indices = np.random.randint(0, X_train.shape[0], acc_check_size)
	y_check = np.zeros([acc_check_size, 2 * y_train.shape[1]])
	y_check[:, :y_train.shape[1]] = y_train[check_indices] # True images
	preds = discriminator.predict(X_train[check_indices])
	acc_real = accuracy(preds, y_check)

	# Check accuracy of discriminator on generated images
	noise = np.random.uniform(0, 1, size=[acc_check_size, input_noise_size])
	noise_class = np.zeros([acc_check_size, y_train.shape[1]])
	for image_class in range(y_train.shape[1]):
		start_row = image_class * acc_check_size / y_train.shape[1]
		end_row = (image_class + 1) * acc_check_size / y_train.shape[1]
		noise_class[start_row:end_row, image_class] = 1
	noise_check = np.concatenate((noise, noise_class), axis=1)

	generated_images = generator.predict(noise_check)

	y_check = np.zeros([acc_check_size, 2 * y_train.shape[1]])
	for image_class in range(y_train.shape[1]):
		start_row = image_class * acc_check_size / y_train.shape[1]
		end_row = (image_class + 1) * acc_check_size / y_train.shape[1]
		y_check[start_row:end_row, y_train.shape[1] + image_class] = 1

	preds = discriminator.predict(generated_images)
	acc_gen = accuracy(preds, y_check)

	# Log progress
	record["disc"].append(d_loss)
	record["gen"].append(g_loss)
	record['acc_real'].append(acc_real)
	record['acc_gen'].append(acc_gen)
	record['gen_batches'].append(g_batch)
	
	# Display progress:
	if epoch % display_interval == 0:
		msg = ""
		msg += "Epoch: {:d}, d_loss: {:.2f}, g_loss: {:.2f}".format(epoch, d_loss, g_loss)
		msg += ", real: {:.2f}, gen: {:.2f}".format(acc_real, acc_gen)
		print msg
		msg = ""
		msg += "Time elapsed: {:.2f} minutes".format((time.time() - start) / 60)
		msg += ", generator batch count: {:.2f} of {:.2f}".format(g_batch, allowed_batches)
		print msg

	# Save progress to file
	if (epoch + 1) % save_interval == 0 and (epoch + 1) != nb_epoch:
		gen_file = "gen-" + str(epoch + 1) + ".h5"
		generator.save(gen_file)
		print "\nGenerator model saved to", gen_file
		disc_file = "disc-" + str(epoch + 1) + ".h5"
		discriminator.save(disc_file)
		print "Discriminator model saved to", disc_file
		pickle_file = "record-" + str(epoch + 1) + ".pickle"
		saveRecords(pickle_file)
		print "Loss records saved to", pickle_file, "\n"

print "Training completed in", (time.time() - start) / 60., "minutes."

print "Generator loss:", record["gen"][-1]
print "Discriminator loss:", record["disc"][-1]
print "Mean accuracy on real images:", np.mean(record["acc_real"][-int(0.1 * nb_epoch):])
print "Mean accuracy on generated images:", np.mean(record["acc_gen"][-int(0.1 * nb_epoch):])

gen_file = "gen-" + str(nb_epoch) + ".h5"
disc_file = "disc-" + str(nb_epoch) + ".h5"
generator.save(gen_file)
discriminator.save(disc_file)

pickle_file = "record-" + str(nb_epoch) + ".pickle"
saveRecords(pickle_file)
statinfo = os.stat(pickle_file)
print 'Compressed pickle size:', statinfo.st_size
