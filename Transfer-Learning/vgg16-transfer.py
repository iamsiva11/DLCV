# Weights File [vgg16_weights.h5]
# https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view

# VGG-16 for Transfer Learning

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam

def vgg16_model():
	model = Sequential()
	model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))
	return model

model = vgg16_model()

# Load weigts from file
model.load_weights('vgg16_weights.h5')

# Remove the last 2 layers 
for l in model.layers[:-2]:
   l.trainable = False
model.layers.pop()
model.layers.pop()

model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer = Adam(), loss='categorical_crossentropy', metrics=['accuracy'])