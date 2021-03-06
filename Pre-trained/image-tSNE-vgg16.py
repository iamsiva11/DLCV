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

# vgg-16 Weights file (~550MB)
# https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view

""" Alternate weights file(in case of issue with he first)
Vggnet weights issue - functional code
ERROR: You are trying to load a weight file containing 37 layers into a model with 38 layers.
http://files.heuritech.com/weights/vgg16_weights.h5
"""


vgg_path = '../data/vgg16_weights.h5'


def get_image_by_url(path):
    fd = urllib.urlopen(image_url)
    image_file = io.BytesIO(fd.read())
    img = Image.open(image_file)
    return img

def get_image(path, isurlpath=False):
	"""convert the image to a numpy array of the correct size for further processing."""
	if isurlpath:
        fd = urllib.urlopen(image_url)
        image_file = io.BytesIO(fd.read())
        img = Image.open(image_file)
    else:
        img = Image.open(path)

	if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)  # resize the image to fit into VGG-16
    img = np.array(img.getdata(), np.uint8) # Convert to numpy array
    img = img.reshape(224, 224, 3).astype(np.float32) # reshape and convert to float32
    # Mean Subtraction
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


# We will load all the paths to our images into an array images,
# recursively from image_path.
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) \
         for f in filenames if os.path.splitext(f)[1].lower() \
         in ['.jpg','.png','.jpeg']]

# If num_images < the number of images you have,
# it will filter out a random subsample of num_images images.
if num_images < len(images):
    images = [images[i] for i in sorted(random.sample(xrange(len(images)), num_images))]

print("keeping %d images to analyze" % len(images))


# Now we will go through every image, and extract its activations. 
# The activations we are interested in are those in the last 
# fully-connected layer of a convnet that we forward pass the image through.

# Each set of activations is a 4096-element list which provides 
# a high-level characterization of that image. Some of the elements 
# may be interpretable, corresponding to real-world objects,
# while others are more abstract. Let's plot it.

activations = []
for idx,image_path in enumerate(images):
    if idx%100==0:
        print "getting activations for %d/%d %s" % (idx+1, len(images), image_path)
    image = get_image(image_path);
    acts = model.predict(image)[0]
    activations.append(acts)

""" It is usually a good idea to first run the vectors through a faster dimensionality 
reduction technique like principal component analysis before using t-SNE, 
to project the 4096-bit activation vectors to a smaller size, say 300 dimensions.
Then run t-SNE over the resulting 300-dimensional vectors to get our final 2-d embedding """

# First run our activations through PCA to get the activations down to 300 dimensions
activations = np.array(activations)
pca = PCA(n_components=300)
pca.fit(activations)
pca_activations = pca.transform(activations)

# Then run the PCA-projected activations through t-SNE to get our final embedding
X = np.array(pca_activations)
tsne = TSNE(n_components=2, 
            learning_rate=150, 
            perplexity=30, 
            verbose=2, 
            angle=0.2).fit_transform(X)


# Now let's also normalize the t-SNE so all values are between 0 and 1.
# normalize t-sne points to {0,1}
tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

# Finally, we will compose a new RGB image where the set of images 
# have been drawn according to the t-SNE results.

width = 3000
height = 3000
max_dim = 100

full_image = Image.new('RGB', (width, height))
for img, x, y in zip(images, tx, ty):
    tile = Image.open(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((tile.width/rs, tile.height/rs), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)))

matplotlib.pyplot.figure(figsize = (12,12))
imshow(full_image)

#You can save the t-SNE image to disk.
full_image.save("myTSNE.png")