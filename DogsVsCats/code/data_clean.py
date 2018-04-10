#from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

cats_index = ['n02123045','n02123159','n02123394',
       'n02123597','n02124075','n02125311','n02127052']

dogs_index = ['n02085620','n02085782','n02085936','n02086079'
,'n02086240','n02086646','n02086910','n02087046'
,'n02087394','n02088094','n02088238','n02088364'
,'n02088466','n02088632','n02089078','n02089867'
,'n02089973','n02090379','n02090622','n02090721'
,'n02091032','n02091134','n02091244','n02091467'
,'n02091635','n02091831','n02092002','n02092339'
,'n02093256','n02093428','n02093647','n02093754'
,'n02093859','n02093991','n02094114','n02094258'
,'n02094433','n02095314','n02095570','n02095889'
,'n02096051','n02096177','n02096294','n02096437'
,'n02096585','n02097047','n02097130','n02097209'
,'n02097298','n02097474','n02097658','n02098105'
,'n02098286','n02098413','n02099267','n02099429'
,'n02099601','n02099712','n02099849','n02100236'
,'n02100583','n02100735','n02100877','n02101006'
,'n02101388','n02101556','n02102040','n02102177'
,'n02102318','n02102480','n02102973','n02104029'
,'n02104365','n02105056','n02105162','n02105251'
,'n02105412','n02105505','n02105641','n02105855'
,'n02106030','n02106166','n02106382','n02106550'
,'n02106662','n02107142','n02107312','n02107574'
,'n02107683','n02107908','n02108000','n02108089'
,'n02108422','n02108551','n02108915','n02109047'
,'n02109525','n02109961','n02110063','n02110185'
,'n02110341','n02110627','n02110806','n02110958'
,'n02111129','n02111277','n02111500','n02111889'
,'n02112018','n02112137','n02112350','n02112706'
,'n02113023','n02113186','n02113624','n02113712'
,'n02113799','n02113978']

cat_img_amount = 12500
dog_img_amount = 12500

def cat_generator(width, height, preprocess, batch_size = 25):
	cat_img = np.zeros([batch_size, width, height, 3])
	cnt = 0
	for i in range(cat_img_amount):
		cat_img[cnt] = image.img_to_array(image.load_img('./train/cat.'+str(i)+'.jpg', target_size=(width,height)))
		cnt += 1

		if cnt == batch_size:
			cnt = 0
			yield preprocess(cat_img)
			#yield cat_img

def dog_generator(width, height, preprocess, batch_size = 25):
	dog_img = np.zeros([batch_size, width, height, 3])
	cnt = 0
	for i in range(dog_img_amount):
		dog_img[cnt] = image.img_to_array(image.load_img('./train/dog.'+str(i)+'.jpg', target_size=(width,height)))
		cnt += 1

		if cnt == batch_size:
			cnt = 0
			yield preprocess(dog_img)


#use resnet to get the img which is not dogs or cats
def get_dirty_img(model, width, height, top_num, preprocess, decode_func):
    dirty_img_index = []
    '''
    cat_img = np.zeros([cat_img_amount, width, height, 3])
    for i in range(cat_img_amount):
    	cat_img[i] = image.img_to_array(image.load_img('./train/cat.'+str(i)+'.jpg', target_size=(width,height)))

    cat_predict = model.predict(preprocess(cat_img), batch_size=32)
    dirty_img_index.append(dirty_judge(cat_predict, 'cat', top_num, decode_func))
    return dirty_img_index[0]#+dirty_img_index[1]
    #resnet_model = ResNet50(weights = 'imagenet')
    '''
    '''
    img_gen = image.ImageDataGenerator(preprocessing_function=preprocess)
    cat_gen = img_gen.flow_from_directory('./train_data/for_gener/cat',target_size=(width, height),
    	shuffle = False,seed=2018)
    dog_gen = img_gen.flow_from_directory('./train_data/for_gener/dog',target_size=(width, height),
    	shuffle = False,seed=2018)
    '''
    cat_predict = []
    dog_predict = []
    cat_gen = cat_generator(width, height, preprocess)
    dog_gen = dog_generator(width, height, preprocess)
    #cat_predict = model.predict_generator(cat_gen,steps = 1)
    #dog_predict = model.predict_generator(dog_gen,steps = 1)
    '''
    cat_predict.append(model.predict_on_batch(next(cat_gen)))
    dog_predict.append(model.predict_on_batch(next(dog_gen)))
    cat_predict.append(model.predict_on_batch(next(cat_gen)))
    dog_predict.append(model.predict_on_batch(next(dog_gen)))
    '''
    for item in cat_gen:
    	cat_predict.append(model.predict_on_batch(item))

    for item in dog_gen:
    	dog_predict.append(model.predict_on_batch(item))

    dirty_img_index.append(dirty_judge(np.concatenate(cat_predict), 'cat', top_num, decode_func))
    dirty_img_index.append(dirty_judge(np.concatenate(dog_predict), 'dog', top_num, decode_func))

    return dirty_img_index[0]+dirty_img_index[1]

    


def dirty_judge(prediction, catalog, top_num, decode_func):
	'''
	img_gen = image.ImageDataGenerator()
	ret = []
	cata = decode_func(prediction, top=top_num)
	if 'cat' == catalog:
		gen = img_gen.flow_from_directory('./train_data/for_gener/cat',
    		shuffle = False,seed=2018)
		principle_index = cats_index
	if 'dog' == catalog:
		principle_index = dogs_index
		gen = img_gen.flow_from_directory('./train_data/for_gener/dog',
			shuffle = False,seed=2018)

	for i,filename in enumerate(gen.filenames):
		pred = cata[i-1]
		if 'cat/cat.1004.jpg' == filename:
			print(pred)
		is_dirty = True
		for j in pred:
			if j[0] in principle_index:
				is_dirty = False
				break

		if is_dirty:
			ret.append(filename)

		if (30*32 == i):
			break

	return ret
	'''

	
	ret = []
	for top in top_num:
		tmp = []
		cata = decode_func(prediction, top=top)
		principle_index = cats_index
		if 'dog' == catalog:
			principle_index = dogs_index

		#print(cata)

		for i in range(prediction.shape[0]):
			is_dirty = True
			for j in range(top):
				if cata[i][j][0] in principle_index:
					is_dirty = False
					break

			if is_dirty:
				tmp.append(catalog+'.'+str(i)+'.jpg')
		ret.append(tmp)
	

	return ret