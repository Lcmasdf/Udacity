from keras.applications.resnet50 import ResNet50
from keras.applications import inception_v3
from keras.applications import xception
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalAveragePooling2D, Input, Lambda
import numpy as np
from keras import Model
import matplotlib.pyplot as plt
import shutil

import h5py
import os

def get_feature(model_name='ResNet50', img_size=(224,224)):

	print('model {} begin feature extract'.format(model_name))

	img_width = img_size[0]
	img_height = img_size[1]
	input_tensor = Input((img_width, img_height, 3))
	x = input_tensor

	if 'ResNet50' == model_name:
		base_model = ResNet50(input_tensor=x, weights='imagenet', include_top=False)
	elif 'InceptionV3' == model_name:
		x = Lambda(inception_v3.preprocess_input)(x)
		base_model = inception_v3.InceptionV3(input_tensor=x, weights='imagenet', include_top=False)
	elif 'Xception' == model_name:
		x = Lambda(xception.preprocess_input)(x)
		base_model = xception.Xception(input_tensor=x, weights='imagenet', include_top=False)
	model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

	datagen = image.ImageDataGenerator()
	train_gener = datagen.flow_from_directory('./train_data',target_size=(img_width,img_height),seed=2018,batch_size=32,class_mode='binary',shuffle=False)
	val_gener = datagen.flow_from_directory('./val_data',target_size=(img_width,img_height),seed=2018,batch_size=32,class_mode='binary',shuffle=False)
	test_gener = datagen.flow_from_directory('./test_data',target_size=(img_width,img_height),seed=2018,batch_size=32,shuffle=False)

	train_feature = model.predict_generator(train_gener)
	val_feature = model.predict_generator(val_gener)
	test_feature = model.predict_generator(test_gener)

	#shutil.rmtree(model_name+'_feature.h5')
	if os.path.exists('./model/feature/'+model_name+'_feature.h5'):
		#shutil.rmtree('./'+model_name+'_feature.h5')
		os.remove('./model/feature/'+model_name+'_feature.h5')
	with h5py.File('./model/feature/'+model_name+'_feature.h5') as h:
		h.create_dataset('train_feature', data = train_feature)
		h.create_dataset('train_label', data = train_gener.classes)
		h.create_dataset('val_feature', data = val_feature)
		h.create_dataset('val_label', data = val_gener.classes)
		h.create_dataset('test_feature', data = test_feature)