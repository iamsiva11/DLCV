"""
Generate a t-SNE of a set of images, using a feature vector 
for each image derived from the activations of the last fully-connected layer 
in a pre-trained convolutional neural network (convnet).
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD

from PIL import Image


# vgg-16 Weigths file (~550MB)
# https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view


vgg_path = '../data/vgg16_weights.h5'


def get_image(path):
	"""convert the image to a numpy array of the correct size for further processing."""
	img = Image.open(path)
	if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)  # resize the image to fit into VGG-16
    img = np.array(img.getdata(), np.uint8) # Convert to numpy array
    img = img.reshape(224, 224, 3).astype(np.float32) # reshape and convert to float32
    # subtract mean (probably unnecessary for t-SNE but good practice)
    img[:,:,0] -= 123.68 
    img[:,:,1] -= 116.779
    img[:,:,2] -= 103.939
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)
    return img





def VGG_16(weights_path=None):
"""
Define the VGG16 neural net architecture and 
load the weights into it from the h5 file downloaded earlier.
"""
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))

    # Load the vgg-16 weights file
    f = h5py.File(weights_path)
    # Assigning appropriate weights from the vgg-16 file

    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    print("finished loading VGGNet")

    return model


model = VGG_16(vgg_path)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')