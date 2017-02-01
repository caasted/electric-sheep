from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, UpSampling2D, Flatten
from keras.layers import Input, Convolution2D, Reshape, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import Adam, Nadam
import numpy as np
from six.moves import cPickle as pickle
import os
import time

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

# Limit data to a single class of images
X_new = np.ndarray(shape=(X_train.shape[0] / 10, X_train.shape[1], 
					X_train.shape[2], X_train.shape[3]), dtype=np.float32)
classType = 3 # Select the types of images to use
sampleNumber = 0
for X, y in zip(X_train, y_train):
	if y[0] == classType:
		X_new[sampleNumber] = X
		sampleNumber += 1
X_train = X_new[:sampleNumber, :, :, :]

print X_train.shape

# Models and training process adapted from 
# https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN_v2.ipynb
# with additional insights from 
# https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_cifar_minibatch_discrimination.py
# and 
# http://torch.ch/blog/2015/11/13/gan.html

# Optimizers
GAN_optimizer = Adam(lr=3e-4)
disc_optimizer = Adam(lr=3e-4)
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
# disc_file = "disc3-1-pre.h5"
# if os.path.isfile(disc_file):
# 	discriminator.load_weights(disc_file)
# 	print "Previously trained discriminator weights loaded."
# else:
noise = np.random.uniform(0, 1, size=[X_train.shape[0], 100])
generated_images = generator.predict(noise)
X_new = np.concatenate((X_train, generated_images))
y_new = np.zeros([2*X_train.shape[0],2])
y_new[:X_train.shape[0],1] = 1
y_new[X_train.shape[0]:,0] = 1

training_lock(discriminator, True)
discriminator.fit(X_new, y_new, nb_epoch=2, batch_size=32)
preds = discriminator.predict(X_new)

preds_class = np.argmax(preds, axis=1)
y_new_class = np.argmax(y_new, axis=1)
accuracy = 1. * (y_new_class == preds_class).sum() / y_new_class.shape[0]
print "Accuracy:", accuracy
# discriminator.save_weights(disc_file)

losses = {"d": [], "g": [], "acc": []}
def train_for_n(nb_epoch=1000, batch_size=100):
	start = time.time()
	for epoch in range(nb_epoch):
		# Make generative images
		image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size), :, :, :]    
		noise = np.random.uniform(0, 1, size=[batch_size, 100])
		generated_images = generator.predict(noise)
		
		# Train discriminator
		X_new = np.concatenate((image_batch, generated_images))
		y_new = np.zeros([2 * batch_size, 2])
		y_new[0:batch_size, 1] = 1
		y_new[batch_size:, 0] = 1
		
		training_lock(discriminator, True)
		d_loss  = discriminator.train_on_batch(X_new, y_new)
		losses["d"].append(d_loss)
	
		# Train GAN
		noise_tr = np.random.uniform(0, 1, size=[batch_size, 100])
		y_new_2 = np.zeros([batch_size,2])
		y_new_2[:,1] = 1
		
		training_lock(discriminator, False)
		g_loss = GAN.train_on_batch(noise_tr, y_new_2 )
		losses["g"].append(g_loss)

		# Check accuracy of discriminator after GAN training
		preds = discriminator.predict(X_new)
		preds_class = np.argmax(preds, axis=1)
		y_class = np.argmax(y_new, axis=1)
		accuracy = 1. * (y_class == preds_class).sum() / y_class.shape[0]
		losses['acc'].append(accuracy)

		if epoch % 10 == 0 and epoch != 0:
			# Report progress:
			print "Epoch:", epoch, ", d_loss:", d_loss, ", g_loss:", g_loss, "Accuracy:", accuracy
			print "Time Remaining:", (nb_epoch - (epoch + 1)) * (time.time() - start) / (60 * (epoch + 1)), "minutes"
		if epoch % 10000 == 0 and epoch != 0:
			gen_file = "gen" + str(classType) + "-" + str(epoch) + ".h5"
			generator.save(gen_file)
			print "\nGenerator model saved to", gen_file, "\n"

# Faces took 100 epochs on a 13,000 dataset -> 13,000 epochs with batch_size = 100?
# They say it took a day training on a GPU for 100 epochs

# Improved GAN (Goodfellow) uses batch_size = 100, loops on dataset / batch_size, then does 1200 epochs...
# That's actually 1200 * 50,000 / 100 = 600,000 of my nb_epoch, grabbing a sample every 500 epochs
# That's about 48 hours of training on a 980ti based on my first success network

# 60,000 batches x 100 samples / batch ~= 2.5 hours
print "Starting training."
nb_epoch = 60000
train_for_n(nb_epoch=nb_epoch, batch_size=100)

gen_file = "gen" + str(classType) + "-" + str(nb_epoch) + ".h5"
disc_file = "disc" + str(classType) + "-" + str(nb_epoch) + ".h5"
generator.save(gen_file)
discriminator.save(disc_file)

print "Training complete."

print "Generator loss:", losses["g"][-1]
print "Discriminator loss:", losses["d"][-1]
print "Discriminator accuracy:", losses["acc"][-1]

pickle_file = "losses" + str(classType) + "-" + str(nb_epoch) + ".pickle"
try:
	f = open(pickle_file, 'wb')
	pickle.dump(losses, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print 'Unable to save data to', pickle_file, ':', e
	raise
	
statinfo = os.stat(pickle_file)
print 'Compressed pickle size:', statinfo.st_size
