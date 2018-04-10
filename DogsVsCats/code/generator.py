import os
import random
import shutil
import tqdm
import time

origin_train_data_path = './train'
train_val_ratio = 0.8

def rmrf_mkdir(dirname):
	if os.path.exists(dirname):
		shutil.rmtree(dirname)
	os.mkdir(dirname)

def create_dir_for_keras_img_generator(dirty_data_index):

	rmrf_mkdir('./train_data')
	os.mkdir('./train_data/cat')
	os.mkdir('./train_data/dog')

	rmrf_mkdir('./val_data')
	os.mkdir('./val_data/cat')
	os.mkdir('./val_data/dog')

	rmrf_mkdir('./test_data')
	os.symlink('../test', './test_data/test')

	for root, dirs, files in os.walk(origin_train_data_path):
		random.shuffle(files)

		total_num = len(files)
		train_files = files[:int(total_num*train_val_ratio)]
		val_files = files[int(total_num*train_val_ratio):]

		for file in train_files:
			time.sleep(0.005)

			if file in dirty_data_index:
				continue

			if file[:3] == 'cat':
				os.symlink('../../train/'+file, './train_data/cat/'+file)
			elif file[:3] == 'dog':
				os.symlink('../../train/'+file, './train_data/dog/'+file)

		for file in val_files:
			if file in dirty_data_index:
				continue

			if file[:3] == 'cat':
				os.symlink('../../train/'+file, './val_data/cat/'+file)
			elif file[:3] == 'dog':
				os.symlink('../../train/'+file, './val_data/dog/'+file)