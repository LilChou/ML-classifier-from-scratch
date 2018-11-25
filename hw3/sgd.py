#! /usr/bin/python

# @author Yu-Jung Chou
# @date Nov 2018

import sys
import functions
import numpy as np


class StochasticGradientDescent():
	def __init__(self, n, regularization, mbs, learning_rate=0.001):
		self.r = learning_rate
		self.reg = regularization
		self.lamb = 0.01 if regularization else 0
		self.mini_batch_size = mbs
		self.weights = np.random.rand(10,n)


	def sigmoid(self, z):
		# the sigmoid of z
		return 1 / (1+np.exp(-z))

	def p_norm(self, p, w):
		# L-2 norm: p=2
		return np.sum(np.power(w, p))

	def one_hot(self, y):
		o_hot = np.zeros((len(y), 10))
		for i, one in enumerate(y):
			o_hot[i, one] = 1
		return o_hot

	def loss(self, o_hot, h):
		# one = np.ones(o_hot.shape)
		return -1*np.mean(o_hot*np.log(h+1e-7) + (1-o_hot)*np.log(1-h+1e-7))

	def fit(self, train_data, train_label, start):
		# fit returns loss
		loss = 0
		
		# get from batch
		X = train_data[start:start+self.mini_batch_size]
		y = train_label[start:start+self.mini_batch_size]
		
		# calculate the gradient
		o_hot = self.one_hot(y)
		z = np.dot(X, self.weights.T)
		h = self.sigmoid(z)
		gd = np.dot((o_hot-h).T, X)

		# regularization
		if self.reg:
			gd -= self.lamb * self.weights
			loss += 1/2 * self.lamb * self.p_norm(2, self.weights)

		# weights update
		self.weights += self.r * gd

		# calculate loss
		loss += self.loss(o_hot, h)
		return loss

	def classify(self, X):
		output = np.dot(X, self.weights.T)
		return np.argmax(output, axis=1)

def data_preprocess(train_data, train_label, test_data, test_label, feature_type, train_size=10000):
	# get the small training set
	train_data = train_data[:train_size]
	train_label = train_label[:train_size]

	# different type different train_data
	if feature_type.lower()=="type1":
		train_data = np.divide(train_data, 255.0)
		test_data = np.divide(test_data, 255.0)
	elif feature_type.lower()=="type2":
		train_data = functions.max_pooling_2x2(train_data)
		train_data = np.divide(train_data, 255.0)
		test_data = functions.max_pooling_2x2(test_data)
		test_data = np.divide(test_data, 255.0)
	else:
		print ("Can't identify the feature type!!")
		exit(0)

	# reshape data from 2d to 1d
	train_data = functions.reshape_2d21d(train_data)
	test_data = functions.reshape_2d21d(test_data)

	# add bias
	one = np.ones((len(train_data), 1))
	train_data = np.concatenate((train_data, one), axis=1)
	one = np.ones((len(test_data), 1))
	test_data = np.concatenate((test_data, one), axis=1)
	
	return [train_data, train_label, test_data, test_label]

def main():
	# get arguments
	regularization, feature_type, path = functions.read_argv()
	epochs, mini_batch_size = 200, 1000
	stop_criteria = 0.0001

	# get data
	train_data = functions.read_gz_idx(path+'train-images-idx3-ubyte.gz')
	train_label = functions.read_gz_idx(path+'train-labels-idx1-ubyte.gz')
	test_data = functions.read_gz_idx(path+'t10k-images-idx3-ubyte.gz')
	test_label = functions.read_gz_idx(path+'t10k-labels-idx1-ubyte.gz')

	# data preprocessing
	train_data, train_label, test_data, test_label = data_preprocess(train_data, train_label, test_data, test_label, feature_type)

	# model initialization
	model = StochasticGradientDescent(len(train_data[0]), regularization, mini_batch_size)

	# initialize list for plotting
	accuracy_train = []
	accuracy_test = []

	# start training
	prev_loss = 0	# for stopping criteria
	epoch = epochs	# for plotting
	l = len(train_data)
	for e in range(epochs):
		# shuffle training data if batch
		if e*mini_batch_size == l:
			train_data, train_label = functions.unison_shuffle(train_data, train_label)

		# model fitting
		loss = model.fit(train_data, train_label, (e*mini_batch_size)%l)
		
		# test the accuracy
		acc_train = functions.accuracy(model.classify(train_data), train_label)/100
		acc_test = functions.accuracy(model.classify(test_data), test_label)/100

		# record for plotting
		accuracy_train.append(acc_train)
		accuracy_test.append(acc_test)

		# log
		print ("epoch {0:7d}:\t Train loss: {1:8.4f},\t Train acc: {2:8.4f}, \tTest acc: {3:8.4f}".format(
			(e+1)*mini_batch_size, loss, acc_train, acc_test))

		# stopping criteria
		if np.abs(prev_loss-loss)<stop_criteria:
			epoch = e+1
			break

		prev_loss = loss
		

	print ('End of Train & Test')
	print ('Plotting ... ')

	# plot to graph
	if regularization:	title = 'SGD Regularize '+feature_type
	else:				title = 'SGD '+feature_type
	functions.plot(title, [e*mini_batch_size for e in range(1, epoch+1)], accuracy_train, accuracy_test)

	print ("End Of The Program")



if __name__=="__main__":
	main()


