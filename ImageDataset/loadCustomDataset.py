from scipy import ndimage
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

def img_Preprocess(imDir):
	imgs = os.listdir(imDir)
	num = len(imgs)
	img_list=[]
	big_img_list=[]

	for i in range(num):
	    new_shape =(30,30)
	    img_path = imDir + os.sep + imgs[i]	    
	    img = load_img(img_path, target_size=new_shape)
	    img = np.asarray(img, dtype='float64') / 256.
	    img = img_to_array(img)
	    img_list.append(img)     
	    return img_list

def preprocess_image_new(image_path):
    img = ndimage.imread(image_path)
    img = imresize(img, target_size) 
    img = img.transpose((2,0,1))    
    return img	    