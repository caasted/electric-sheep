from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, UpSampling2D, Flatten
from keras.layers import Input, Convolution2D, Reshape, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import Adam, Nadam, Adadelta
import numpy as np
from six.moves import cPickle as pickle
import os
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
# 10: All
image_class = 1

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

# Convert uint8 pixel values to float32 in the range [-1, 1] (for tanh)
# X_train = X_train.astype('float32')
# X_train /= 128
# X_train -= 128

# Option to limit data to a single class of images
if image_class < 10:
	X_new = np.ndarray(shape=(X_train.shape[0] / 10, X_train.shape[1], 
						X_train.shape[2], X_train.shape[3]), dtype=np.float32)
	sampleNumber = 0
	for X, y in zip(X_train, y_train):
		if y[0] == image_class:
			X_new[sampleNumber] = X
			sampleNumber += 1
	X_train = X_new[:sampleNumber, :, :, :]

print X_train.shape

# Models and training process adapted from the following three sources:
# https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN_v2.ipynb
# https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_cifar_minibatch_discrimination.py
# http://torch.ch/blog/2015/11/13/gan.html

# Optimizers
# GAN_optimizer = Adam(lr=3e-4)
# disc_optimizer = Adam(lr=3e-4)
GAN_optimizer = Adadelta()						# Currently gen loss increases and disc loss decreases over time
disc_optimizer = Adadelta()						# Need to strengthen gen or weaken disc
dropout_rate = 0.25

# Generator Model
g_input = Input(shape=[100])
g_layer = Dense(512*4*4)(g_input)
g_layer = BatchNormalization(mode=2)(g_layer)
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
g_layer = Convolution2D(64, 5, 5, border_mode='same')(g_layer)
g_layer = BatchNormalization(mode=2)(g_layer)
g_layer = Activation('relu')(g_layer)

g_layer = Convolution2D(3, 3, 3, border_mode='same')(g_layer)
g_output = Activation('sigmoid')(g_layer)

generator = Model(g_input, g_output)
generator.compile(loss='binary_crossentropy', optimizer=GAN_optimizer)
# generator.summary()

# Discriminator Model
d_input = Input(shape=X_train.shape[1:])

d_layer = Convolution2D(32, 5, 5, border_mode='same')(d_input)
d_layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(d_layer)
d_layer = Activation('relu')(d_layer)
d_layer = Dropout(dropout_rate)(d_layer)

d_layer = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same')(d_layer) # 32 -> 16
d_layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(d_layer)
d_layer = Activation('relu')(d_layer)
d_layer = Dropout(dropout_rate)(d_layer)

d_layer = Convolution2D(96, 5, 5, border_mode='same')(d_layer)
d_layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(d_layer)
d_layer = Activation('relu')(d_layer)
d_layer = Dropout(dropout_rate)(d_layer)

d_layer = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode='same')(d_layer) # 16 -> 8
d_layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(d_layer)
d_layer = Activation('relu')(d_layer)
d_layer = Dropout(dropout_rate)(d_layer)

d_layer = Convolution2D(160, 5, 5, border_mode='same')(d_layer)
d_layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(d_layer)
d_layer = Activation('relu')(d_layer)
d_layer = Dropout(dropout_rate)(d_layer)

d_layer = Convolution2D(192, 5, 5, subsample=(2, 2), border_mode='same')(d_layer) # 8 -> 4
d_layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(d_layer)
d_layer = Activation('relu')(d_layer)
d_layer = Dropout(dropout_rate)(d_layer)

d_layer = Flatten()(d_layer)

d_layer = Dense(1024)(d_layer)
d_layer = Activation('relu')(d_layer)
d_layer = Dropout(dropout_rate)(d_layer)

d_output = Dense(2, activation='softmax')(d_layer)

discriminator = Model(d_input, d_output)
discriminator.compile(loss='categorical_crossentropy', optimizer=disc_optimizer)
# discriminator.summary()

# GAN training helper function
def training_lock(model, setting = False):
	model.trainable = setting
	for layer in model.layers:
		layer.trainable = setting

# Lock weights in the discriminator for training generator
training_lock(discriminator, False)

# GAN Model
gan_input = Input(shape=[100])
gan_layer = generator(gan_input)
gan_output = discriminator(gan_layer)
GAN = Model(gan_input, gan_output)
GAN.compile(loss='categorical_crossentropy', optimizer=GAN_optimizer)
# GAN.summary()

# Pre-train discriminator
noise = np.random.uniform(0, 1, size=[X_train.shape[0], 100])
generated_images = generator.predict(noise)
X_pre = np.concatenate((X_train, generated_images))
y_pre = np.zeros([2*X_train.shape[0],2])
y_pre[:X_train.shape[0],1] = 1
y_pre[X_train.shape[0]:,0] = 1

training_lock(discriminator, True)
discriminator.fit(X_pre, y_pre, nb_epoch=1, batch_size=32)
preds = discriminator.predict(X_pre)

preds_class = np.argmax(preds, axis=1)
y_new_class = np.argmax(y_pre, axis=1)
accuracy = 1. * (y_new_class == preds_class).sum() / y_new_class.shape[0]
print "Accuracy:", accuracy

record = {"disc": [], "gen": [], "acc": []}
def train_for_n(nb_epoch=1200, batch_size=100):
	
	start = time.time()
	batches_per_epoch = X_train.shape[0] / batch_size
	for epoch in range(nb_epoch):
		
		d_loss = 0
		g_loss = 0
		# Loop through all samples each epoch
		for batch_number in range(batches_per_epoch):
			
			# Section of training data to use
			batch_start = batch_number * batch_size
			batch_end = (batch_number + 1) * batch_size
			
			# Make generative images
			image_batch = X_train[batch_start:batch_end, :, :, :]    
			noise_batch = np.random.uniform(0, 1, size=[batch_size, 100])
			generated_images = generator.predict(noise_batch)
			
			# Train discriminator
			X_batch = np.concatenate((image_batch, generated_images))
			y_batch = np.zeros([2 * batch_size, 2])
			y_batch[0:batch_size, 1] = 1
			y_batch[batch_size:, 0] = 1
			
			training_lock(discriminator, True) # Set layers to trainable
			d_loss  += discriminator.train_on_batch(X_batch, y_batch)
			training_lock(discriminator, False) # Set layers to not trainable
			
			# Train GAN
			noise_batch = np.random.uniform(0, 1, size=[batch_size, 100]) # New random inputs
			y_batch_2 = np.zeros([batch_size, 2])
			y_batch_2[:, 1] = 1
			
			g_loss += GAN.train_on_batch(noise_batch, y_batch_2)
			
		# Check accuracy of discriminator after epoch
		noise = np.random.uniform(0, 1, size=[X_train.shape[0], 100])
		generated_images = generator.predict(noise)
		X_check = np.concatenate((X_train, generated_images))
		y_check = np.zeros([2*X_train.shape[0], 2])
		y_check[:X_train.shape[0], 1] = 1
		y_check[X_train.shape[0]:, 0] = 1
		
		preds = discriminator.predict(X_check)
		preds_class = np.argmax(preds, axis=1)
		y_class = np.argmax(y_check, axis=1)
		accuracy = 1. * (y_class == preds_class).sum() / y_class.shape[0]
		
		d_loss /= batches_per_epoch
		g_loss /= batches_per_epoch

		# Log progress
		record['acc'].append(accuracy)
		record["disc"].append(d_loss)
		record["gen"].append(g_loss)

		# Report progress:
		print "Epoch:", epoch, ", d_loss:", d_loss, ", g_loss:", g_loss, "Accuracy:", accuracy
		print "Time Remaining:", (nb_epoch - (epoch + 1)) * (time.time() - start) / (60 * (epoch + 1)), "minutes"

		if epoch % (nb_epoch / 10) == 0 and epoch != 0:
			gen_file = "gen" + str(image_class) + "-" + str(epoch) + ".h5"
			generator.save(gen_file)
			print "\nGenerator model saved to", gen_file, "\n"

print "Starting training."
nb_epoch = 1200
train_for_n(nb_epoch=nb_epoch, batch_size=100)

gen_file = "gen" + str(image_class) + "-" + str(nb_epoch) + ".h5"
disc_file = "disc" + str(image_class) + "-" + str(nb_epoch) + ".h5"
generator.save(gen_file)
discriminator.save(disc_file)

print "Training complete."

print "Generator loss:", record["gen"][-1]
print "Discriminator loss:", record["disc"][-1]
print "Discriminator accuracy:", record["acc"][-1]

pickle_file = "record" + str(image_class) + "-" + str(nb_epoch) + ".pickle"
try:
	f = open(pickle_file, 'wb')
	pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print 'Unable to save data to', pickle_file, ':', e
	raise
	
statinfo = os.stat(pickle_file)
print 'Compressed pickle size:', statinfo.st_size
