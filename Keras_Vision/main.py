# Source for tutorial and code https://elitedatascience.com/keras-tutorial-deep-learning-in-python

import numpy as np
import theano
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
np.random.seed(123)
from keras.datasets import mnist
import matplotlib
from matplotlib import pyplot as plt
#Load shuffled MNIST data

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#Prints (60000, 28, 28)  60000 samples, 28x28 pixels
print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
print(X_train.shape)

X_train = X_train.astype('float32')
X_test= X_test.astype('float32')
# Normalize data
X_train /= 255
X_test /= 255

# Split data to distinct label class
Y_test = np_utils.to_categorical(Y_test, 10)
Y_train = np_utils.to_categorical(Y_train, 10)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
model.add(Convolution2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)