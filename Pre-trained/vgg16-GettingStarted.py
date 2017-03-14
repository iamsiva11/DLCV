from keras.applications import VGG16

vgg_model = VGG16(weights='imagenet', include_top=True)

#print vgg_model.summary()

# Print layer 
for i, layer in enumerate(vgg_model.layers):
    print (i,layer.name, layer.output_shape)