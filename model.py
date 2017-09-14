#this python modul trains the deep neural network based on command line arguments
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
import scipy.misc
import tensorflow as tf
import sys


#training data (with augmentation) was prepared before to separate CPU/hard-drive intensive rotation/flipping/saving operations and GPU-intensive operations (training model)
#each of prepared augmentation training sets has its own driving_log file
def read_csv(delta=0.2, rotation=0.0):
	samples = []
	with open('driving_log_'+str(rotation)+'.csv') as csvfile:
		reader = csv.reader(csvfile)
		for sample in reader:
			samples.append(sample)
	return samples


#this function read files from drive in batches to save memory
def generator(samples, batch_size=4, delta = 0.2, rotation = 0.0):
		num_samples = len(samples)
		while 1: 
			samples = sklearn.utils.shuffle(samples)
			for offset in range(0, num_samples, batch_size):
				if offset+batch_size<num_samples:
					batch_samples = samples[offset:offset+batch_size]
				else:
					batch_samples = samples[offset:num_samples]
				images = []
				angles = []
				for batch_sample in batch_samples:
					angle = float(batch_sample[1])
					source_path = batch_sample[0]
					filename = source_path.split('\\')[-1]
					delta_angle = 0
					if 'left' in filename:
						delta_angle = delta
					elif 'right' in filename:
						delta_angle = -1*delta
					angle = angle+delta_angle
					current_path = 'IMG_'+str(rotation)+'/' + filename
					image = scipy.misc.imread(current_path)
					images.append(image)
					angles.append(angle)
				X_train = np.array(images)
				y_train = np.array(angles)
				yield (X_train, y_train)


#this functios gets its input parameters from command line
def main(argv):
	delta = 0.2
	rotation = 0.0
	batch_size = 4
	nb_epochs = 3
	if len(argv)==2:
		delta = float(argv[1])
	elif len(argv)==3:
		delta = float(argv[1])
		rotation = float(argv[2])
	elif len(argv)==4:
		delta = float(argv[1])
		rotation = float(argv[2])
		batch_size = int(argv[3])
	elif len(argv)==5:
		delta = float(argv[1])
		rotation = float(argv[2])
		batch_size = int(argv[3])
		nb_epochs = int(argv[4])

	print(40*'_')
	print('delta:%.2f'%(delta))
	print('rotation:%.2f'%(rotation))
	print('batch size:%d'%(batch_size))
	print('nb epochs:%d'%(nb_epochs))
	print(40*'_')

	#since I do not know optimal parameters for data augmentation I need to train model on numerous parameter combinations.
	#it is not clever to change those  parameters manually and then restart the training.
	#thefore I prepared for several delta-rotation combinations data sets, which read_csv function can load based on delta and rotation.
	#in that case "delta" is offset for left (-delta) and right cameras (-delta), rotation is a rotation angle for augmentation data.
	samples = read_csv(delta, rotation)
	train_samples, validation_samples = train_test_split(samples, test_size=0.2)
		
	train_generator = generator(train_samples, batch_size=batch_size, delta = 0.2, rotation = rotation)
	validation_generator = generator(validation_samples, batch_size=batch_size, delta = 0.2, rotation = rotation)

	#ch, row, col = 3, 80, 320 
	# Trimmed image format
	row_low = 25
	row_high = 70

	model = Sequential()
	model.add(Cropping2D(cropping=((row_high,row_low), (0,0)), input_shape=(160,320,3)))
	# Preprocess incoming data, centered around zero with small standard deviation 
	model.add(Lambda(lambda x: (x / 255) - 0.5))

	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))

	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam')

	#history objects are needed to check loss/validation loss improvements over training iterations
	history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
						validation_data=validation_generator, nb_val_samples = len(validation_samples), \
						nb_epoch=nb_epochs, verbose = 1)

	model_name = 'model_'+str(delta)+'_w_rotation_'+str(rotation)+'_loss_%.3g_val_loss_%.3g.h5' % (history_object.history['loss'][nb_epochs-1],history_object.history['val_loss'][nb_epochs-1]) 
	model.save(model_name)
	model.save('m_'+str(rotation) +'.h5')

	### print the keys contained in the history object
	print("History object:")
	for key in history_object.history.keys():
		print(key)
		print("%.3g" % history_object.history[key][0])

	#history objects will be dumped into picles objects
	import pickle

	file_name = 'history_object_'+str(delta)+ '_w_rotation_' + str(rotation)+'_loss_%.3g_val_loss_%.3g.obj' %\
		 (history_object.history['loss'][nb_epochs-1],history_object.history['val_loss'][nb_epochs-1])
	file = open(file_name,"wb")
	pickle.dump(history_object.history,file)
	file.close()

if __name__=='__main__':
	main(sys.argv)
