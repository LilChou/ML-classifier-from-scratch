#! /usr/bin/python

# CS578 Hw2	Fall18 @ Purdue University
# @author Yu-Jung Chou
# @date	Oct. 7 2018

import sys			# to read argv	
import functions	# functions I design in functions.py
import numpy as np

class winnow():

	def __init__(self, n, learning_rate, epochs):
		self.threshold = n 	# number of the features, which is used as threshold theta
		self.r = float(learning_rate)	# set to 2 for regular winnow algo, alpha
		self.e = epochs
		self.perceptron = np.array([np.ones(n) for i in range(10)])
		
	def train(self, data, label):
		for epoch in range(self.e):				# epochs
			for digit in range(10):				# perceptrons 0-9 for diff class
				for i, x in enumerate(data):
					score = np.dot(x, self.perceptron[digit])
					if score >= self.threshold:				# predict = True
						if digit == label[i]:	continue	# classified correctly
						else:
							for j in range(len(x)):
								if x[j]:	self.perceptron[digit][j] /= self.r
					else:									# predict = False
						if digit != label[i]:	continue	# classified correctly
						else:
							for j in range(len(x)):
								if x[j]:	self.perceptron[digit][j] *= self.r					
							
	def print_weights(self):
		for i in range(10):
			print self.sum_perceptron[i]

	def classify(self, data):
		prediction = []
		for row in data:
			cur_score = 0
			cur_class = 0
			for digit in range(10):
				score = np.dot(row, self.perceptron[digit])
				if score > cur_score:	cur_score, cur_class = score, digit
			prediction.append(cur_class)
		return np.array(prediction)

def mnist_data_preprocess(train_data, train_label, test_data, test_label, train_size):
	# only need part of the training set
	train_ind = np.array([i for i in range(len(train_data))])
	np.random.shuffle(train_ind)
	train_ind = train_ind[:train_size]

	# get the small training set
	train_data = functions.train_split(train_data, train_ind)
	train_label = functions.train_split(train_label, train_ind)
	
	# reshape data from 2d to 1d
	train_data = functions.reshape_2d21d(train_data)
	test_data = functions.reshape_2d21d(test_data)
	
	# round the value to make it includes only 0 and 1
	train_data = np.around(train_data/255.0)
	test_data = np.around(test_data/255.0)
	
	# # add bias
	# one = np.ones((len(train_data), 1))
	# train_data = np.concatenate((train_data, one), axis=1)
	# one = np.ones((len(test_data), 1))
	# test_data = np.concatenate((test_data, one), axis=1)
	
	return [train_data, train_label, test_data, test_label]

def main():
	# get arguments
	train_size, epochs, learning_rate, path = functions.read_argv()

	# get data
	train_data = functions.read_gz_idx(path+'train-images-idx3-ubyte.gz')
	train_label = functions.read_gz_idx(path+'train-labels-idx1-ubyte.gz')
	test_data = functions.read_gz_idx(path+'t10k-images-idx3-ubyte.gz')
	test_label = functions.read_gz_idx(path+'t10k-labels-idx1-ubyte.gz')

	train_data, train_label, test_data, test_label = mnist_data_preprocess(train_data, train_label, test_data, test_label, train_size)

	# start training
	model = winnow(len(train_data[0]), learning_rate, epochs)
	model.train(train_data, train_label)
	# model.print_weights()
	
	# classify
	prediction = model.classify(train_data)
	f1, p, r = functions.f1_score(prediction, train_label)
	print ' > train f1 score: %.3f\t' % (f1)
	# print ' > train f1 score: %.3f\tprecision: %.3f\trecall: %.3f' % (f1, p, r)

	prediction = model.classify(test_data)
	f1, p, r = functions.f1_score(prediction, test_label)
	print ' >  test f1 score: %.3f\t' % (f1)
	# print ' >  test f1 score: %.3f\tprecision: %.3f\trecall: %.3f' % (f1, p, r)

	# print 'end of the program'






if __name__=='__main__':
	main()





