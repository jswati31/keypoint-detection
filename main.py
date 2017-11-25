from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Flatten, Dense
from keras import backend as K
import load_data as ld
import matplotlib.pyplot as plt
import argparse
import numpy as np

def model_fkpd(input_dim):
	'''
	Network
	input shape : (height, width, channels)
	output : model
	'''
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=input_dim))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 2, 2, border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(128, 2, 2, border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(500))
	model.add(Activation('sigmoid'))
	model.add(Dense(30, activation='linear'))
	#compiling the model
	model.compile(loss='mse', optimizer='RMSprop', metrics=['acc'])
	return model

def train(model, data, args):
	'''
	Training model
	input : (model, data, arguments)
	output :  trained model
	'''
	(X_train, target_input), (X_val, target_val) = data
	hist = model.fit(X_train, target_input, validation_data=(X_val, target_val), nb_epoch=args.epochs, batch_size=args.batch_size)
	if args.save_weights:
		model.save_weights('model_weights.h5', overwrite=True)
	return hist

def test(model, data):
	'''
	Predict error
	input : (model, data)
	output : error
	'''
	(X_test, true_labels) = data
	#accuracy
	predictions = model.predict(X_test)
	n = X_test.shape[0]
	error=0
	for i in range(n):
	    t = np.transpose([float(item) for item in true_labels[i]])
	    diff = np.square((predictions[i,:] - t))
	    error += np.sum(diff)
	return error


def plot(hist):
	'''
	plot loss and accuracy curves
	'''
	# Plot the Training and Validation Loss
	plt.plot(hist.history['loss'], label='Training Loss')
	plt.plot(hist.history['val_loss'], label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	legend = plt.legend(loc='upper center', shadow=True)
	plt.show()

	# Plot the Training and Validation Accuracy
	plt.plot(hist.history['acc'], label='Training Accuracy')
	plt.plot(hist.history['val_acc'], label='Validation Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	legend = plt.legend(loc='upper center', shadow=True)
	plt.show()


if __name__ == "__main__":

	# setting the hyper parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
	parser.add_argument('--epochs', default=400, type=int, help='Number of epochs')
	parser.add_argument('--is_training', default=1, type=int, help='Training(1) or testing(0)')
	parser.add_argument('--data_path', default='data/',help='Path to data folder')
	parser.add_argument('--save_weights', default=1, type=int, help='Save weights (Yes=1, No=0)')
	parser.add_argument('--plot', default=1, type=int, help='Plot accuracy or loss curves (Yes=1, No=0)')
	args = parser.parse_args()
	
	#load data
	(X_train, target_input), (X_val, target_val), (X_test, true_labels) = ld.load()
	
	image_height, image_width, depth = 96, 96, 1
	input_dim = (image_height,image_width,depth)
	#define model
	model = model_fkpd(input_dim)

	# train or test
	if args.is_training:
		hist = train(model=model, data=((X_train, target_input), (X_val, target_val)), args=args)
		if args.plot:
			plot(hist)
	else:  # as long as weights are given, will run testing
		model.load_weights('model_weights.h5')
		error = test(model=model, data=((X_test, true_labels)))
		print("RMSE Value : ", np.sqrt(error/(30*X_test.shape[0])))

