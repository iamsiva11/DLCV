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

def preprocess_image_1(image_path):
	# Had color issues
    img = ndimage.imread(image_path)
    img = imresize(img, target_size) 
    img = img.transpose((2,0,1))   
    return img	    

def preprocess_image_2(image_path):
	# Works fine
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    return img    


def create_dataset(imdir_class):    
	"""
	imdir_class - directory path of a class
	"""
	img_list=[]
    imgs = sorted(os.listdir(imdir_class))
    num = len(os.listdir(imdir_class))
    for i in range(num):
        img = imdir_class + os.sep + imgs[i]
        x =  preprocess_image_2(img)
        img_list.append(x)        
    return img_list

if __name__ == "__main__":
	imDir_adidas ="path/data/"        
	x_train = create_dataset(imDir_adidas)
	x_train = np.stack(img_list)#.astype("float32")
	x_train /= 255