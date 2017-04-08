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
