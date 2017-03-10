# Importing the dataset from keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

# Data Acquisition
##################
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Image data dimensions(of test and trainign data)
n_train, height, width = X_train.shape
n_test, _ , _ = X_test.shape

"""
We have 60,000 28x28 training grayscale images and 10,000
28x28 test grayscale images.
"""

# Data Preprocessing(Image Data)
################################

# Convert the image data ttype to float32(Currently int8)
# We have to preprocess the data into the right form
X_train = X_train.reshape(n_train, 1, height, width).astype('float32')
X_test  = X_test.reshape(n_test, 1, height, width).astype('float32')

# Normalize from [0, 255] to [0, 1]
X_train /= 255
X_test /= 255

# One Hot encoding of labels
# convert class vectors to binary class matrices
n_classes = 10
Y_train = to_categorical(Y_train, n_classes)
Y_test = to_categorical(Y_test, n_classes)

""" Encodes the Class Labels from
[0] to [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
"""

# Designing the architecture of a CNN
#####################################

"""CNN Model Parameters"""

# number of convolutional filters
n_filters = 32
# convolution kernel/filter size
# i.e. we will use a n_conv x n_conv filter
n_conv = 3
# pooling window size
# i.e. we will use a n_pool x n_pool pooling window
n_pool = 2

model = Sequential()

# 1st Half of the Network architecture

# Convolutional layer - 1 
model.add(Convolution2D(
        n_filters, n_conv, n_conv,
        # apply the filter to only full parts of the image
        # (i.e. do not "spill over" the border)
        # this is called a narrow convolution
        border_mode='valid',
        # we have a 28x28 single channel (grayscale) image
        # so the input shape should be (1, 28, 28)
        input_shape=input_shape #(1, height, width)
))
model.add(Activation('relu'))
#Convolutional layer - 2
model.add(Convolution2D(n_filters, n_conv, n_conv))
model.add(Activation('relu'))
# The particular pooling layer we're using is a max pooling layer, 
# which can be thought of as a "feature detector".
# Then we apply pooling to summarize the features Extracted thus far
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))

# 2nd Half of the Network architecture

# Then we can add dropout and our dense and output (softmax) layers.
model.add(Dropout(0.25))
# Flatten the data for the 1D layers
model.add(Flatten())
# FC Layer
# Dense(n_outputs)
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Final Output layer - most preferably softamax
# The softmax output layer gives us a probablity for each class
model.add(Dense(n_classes))
model.add(Activation('softmax'))

# Model Compilation
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Training the Network
######################

# how many examples to look at during each training iteration
batch_size = 128
# how many times to run through the full set of examples
n_epochs = 2 # For learnign Purposes , keeping it small

# The training may be slow depending on your computer
model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          nb_epoch=n_epochs,
          validation_data = (X_test, Y_test))

# Model Accuracy
################

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])