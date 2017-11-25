import pandas as pd
import numpy as np
from collections import defaultdict
import csv

trainFile = 'data/train.csv'
labelFile = 'data/testlabels.csv'
testFile = 'data/test.csv'

def load():
	#Read data from .csv file
	data = pd.read_csv(trainFile)
	data = data.dropna()
	image_data = data['Image'].apply(lambda im: np.fromstring(im, sep=' '))
	X = np.vstack(image_data.values) / 255.  # scale pixel values to [0, 1]
	X = X.astype(np.float32)
	#Target labels
	output = []
	for i in range(len(data.keys())-1):
	    output.append(data[str(data.keys()[i])])

	target_labels=[]
	for i in range(len(output[0])):
	    target_labels.append(zip(*output)[i])
	target_labels = np.asarray(target_labels)
	image_height, image_width, depth = 96, 96, 1
	#data input
	X_train = []
	for i in range(X.shape[0]):
	    image = X[i,:]
	    X_train.append(image)
	X_train = np.asarray(X_train)
	X_train = X_train.reshape(X_train.shape[0], image_height, image_width, depth)
	X_train = X_train.astype('float32')
	#splitting data into validation and training.
	X_val = X_train[:200,:,:,:]
	X_input = X_train[200:,:,:,:]
	target_val = target_labels[:200,:]
	target_input = target_labels[200:,:]

	# PROCESSING TESTING DATA
	true_labels=[]
	#true labels
	columns=defaultdict(list)
	with open(labelFile) as f:
		reader=csv.reader(f)
		reader.next()
		for row in reader:
			for (i,v) in enumerate(row):
				columns[i].append(v)
	#building our true labels
	output=[]
	for i in range(1,len(columns)):
	    output += [columns[i]]

	for i in range(len(output[0])):
	    true_labels.append(zip(*output)[i])

	test_df = pd.read_csv(testFile)
	test_data = test_df['IMAGE'].apply(lambda im: np.fromstring(im, sep=' '))
	X_1 = np.vstack(test_data.values) / 255.  # scale pixel values to [0, 1]
	X_1 = X_1.astype(np.float32)
	image_height, image_width, depth = 96, 96, 1
	# test inputs
	X_test = []
	for i in range(X_1.shape[0]):
	    image = X_1[i,:]
	    X_test.append(image)
	    
	X_test = np.asarray(X_test)
	X_test = X_test.reshape(X_test.shape[0], image_height, image_width, depth)
	X_test = X_test.astype('float32')
	
	return (X_input, target_input), (X_val, target_val), (X_test, true_labels)
