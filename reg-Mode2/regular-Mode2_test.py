from keras.datasets import cifar10
from keras.models import load_model
import numpy as np
from six.moves import cPickle as pickle

# Fetch data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert uint8 pixel values to float32 in the range [0, 1] (for sigmoid)
X_test = X_test.astype('float32')
X_test /= 255

epochs = range(10, 501, 10)
acc_trend = []
for epoch in epochs:
	all_class_preds = np.ndarray(shape=(X_test.shape[0], 10), dtype=np.float32)
	for image_class in range(10):
		discriminator = load_model("networks/disc" + str(image_class) + "-" + str(epoch) + ".h5")
		preds = discriminator.predict(X_test)
		class_preds = preds[:, 0]
		all_class_preds[:, image_class] = class_preds

	# Calculate joint accuracy
	all_class_preds_index = np.argmax(all_class_preds, axis=1)
	accuracy = 0
	for y, p in zip(y_test, all_class_preds_index):
		if y == p:
			accuracy += 1
	accuracy /= 1. * len(y_test)
	acc_trend.append(accuracy)
	
	print "Epoch: {:d}, Joint Accuracy: {:.4f}".format(epoch, accuracy)

pickle.dump(acc_trend, open("regular-Mode2_acc.pickle", "wb" ))
