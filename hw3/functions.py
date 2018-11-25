#! /usr/bin/env python

# @author Yu-Jung Chou

import struct
import numpy as np
import gzip
import sys

import matplotlib.pyplot as plt

def plot(tit, e, train, test):
	plt.title(tit)

	plt.plot(e, train, color='blue', linestyle='--', label='train')
	plt.plot(e, test, color='green', linestyle='-', label='test')

	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.grid(True)
	plt.legend()

	plt.show()
	# plt.savefig('convergence.png')


def accuracy(predicted_label, true_label):
	# two input should be np.array
	return np.mean(predicted_label==true_label)*100

def f1_score(predicted_label, true_label):
	table = [[0 for i in range(10)] for j in range(10)]
	total = len(predicted_label)
	for i in range(total):
		table[int(true_label[i])][int(predicted_label[i])] += 1

	p_l, r_l, f_l = [], [], []
	for i in range(10):
		TP = float(table[i][i])
		FP, FN = (-1)*TP, (-1)*TP
		for j in range(10):	FP += table[j][i]
		for j in range(10):	FN += table[i][j]
		TN = total-TP-FP-FN

		if TP==0:
			if FP==0 and FN==0:
				# precision, recall, f1_score = 1, 1, 1
				continue
			else:
				precision, recall, f1_score = 0, 0, 0
		else:
			precision = TP/(TP+FP)
			recall = TP/(TP+FN)
			f1_score = 2*precision*recall/(precision+recall)
		p_l.append(precision)
		r_l.append(recall)
		f_l.append(f1_score)

	macro_f1 = sum(f_l)/len(f_l)*100
	macro_p = sum(p_l)/len(p_l)*100
	macro_r = sum(r_l)/len(r_l)*100
	return [macro_f1, macro_p, macro_r]

def unison_shuffle(X, y):
	assert len(X)==len(y)
	i = np.random.permutation(len(X))
	return X[i], y[i]

def max_pooling_2x2(train_data):
	new_data = []
	for data in train_data:
		tmp = [[0 for i in range(len(data[0])/2)] for j in range(len(data)/2)]
		for i in range(0, len(data), 2):
			for j in range(0, len(data[0]), 2):
				max_pool = max(data[i][j], data[i+1][j], data[i+1][j], data[i+1][j+1])
				tmp[i/2][j/2] = max_pool
		new_data.append(tmp)
	return np.array(new_data)


def reshape_2d21d(list_of_pics):
	l = len(list_of_pics[0])*len(list_of_pics[0][0])
	new_list = []
	for pic in list_of_pics:
		new_pic = np.reshape(pic, l)
		new_list.append(new_pic)
	return np.array(new_list)

def train_split(data, index):
	new_data = []
	for i in index:
		new_data.append(data[i])
	return np.array(new_data)

def read_argv():
	if len(sys.argv) != 4:
		print "[X] Error: Usage should be:"
		print "[X]        python program.py [reqularization: True / False] [Feature_type: type1 / type2] [path to data folder]"
		exit(0)

	if sys.argv[1] == 'False':	regularization = False
	elif sys.argv[1]=='True':	regularization = True
	else:
		print ("Can't recognize the argv")
		exit(0)

	feature_type = str(sys.argv[2])
	path = sys.argv[3]
	if path[-1] != '/':	path += '/'

	print ('> Regularization: %r' % regularization),
	print ('\t feature type: %s' % feature_type),
	print ('\t  path: %s' % path)

	return [regularization, feature_type, path]
	# return [train_size, epochs, learning_rate, path]

def read_gz_idx(filename):
	try:
		with gzip.open(filename, 'rb') as f:
			zero, data_type, dims = struct.unpack('>HBB', f.read(4))
			shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
			return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
	except:
		print "[X] can't find the files in the folder"
		exit(0)

def test_read():
	# train data
	train_data = read_idx('train-images-idx3-ubyte.gz')
	train_label = read_idx('train-labels-idx1-ubyte.gz')
	
	train_ind = np.array([i for i in range(60000)])
	np.random.shuffle(train_ind)
	train_ind = train_ind[:10000]
	
	# test data
	test_data = read_idx('t10k-images-idx3-ubyte.gz')
	test_label = read_idx('t10k-labels-idx1-ubyte.gz')
	# for i in range(28):
	# 	for j in range(28):
	# 		# print test_data[0][i][j],
	# 		test_data[0][i][j] = round(test_data[0][i][j]/255)
	# 		print test_data[0][i][j],
	# 	print ''
	# print test_data.shape
	# print test_label.shape
	# print test_data[:5]
	# print test_label[:5]

if __name__=="__main__":
	test_read()