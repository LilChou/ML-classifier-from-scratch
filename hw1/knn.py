#! /urs/bin/env python

import numpy as np
import csv
from math import *

# def chebyshev_distance():
# 	print 'hello chebyshev'
# 	print 'this is the minkowski_distance where the k->infinite'
# def minkowski_distance():
# 	print 'hello minkowski'
# 	print 'but minkowski need a lambda'
# 	print 'lambda=1 	i.e. manhattan distance'
# 	print 'lambda=2 	i.e. euclidean distance'
# 	print 'lambda=inf 	i.e. chebyshev distance'
# def hamming_distance():
# 	print 'hello hamming'
# 	print 'but hamming is mainly used for string difference'
# 	print 'or at least use unsigned int to calculate the XOR bits'
# def mahalanobis_distance():
# 	print 'hello mahalanobis'
# # 	print 'this includes vector computation'
# def split_data_n_label(data_n_label):
# 	print ' >  split data and label'
# 	labels = []
# 	for row in data_n_label:
# 		labels.append(row.pop())
# 	return data_n_label, labels


def distance_measures(pointA, pointB, ind_of_func):
	if ind_of_func==0:	return euclidean_distance(pointA, pointB)
	if ind_of_func==1:	return manhattan_distance(pointA, pointB)
	if ind_of_func==2:	return canberra_distance(pointA, pointB)

def euclidean_distance(pointA, pointB):
	# last col is label
	return sqrt(sum( (a-b)**2 for a,b in zip(pointA, pointB)[:-1] ))
	print 'hello euclidean'
def manhattan_distance(pointA, pointB):
	# last col is label
	return sum( abs(a-b) for a,b in zip(pointA, pointB)[:-1] )
	print 'hello manhattan'
def canberra_distance(pointA, pointB):
	# last col is label
	return sum( abs(a-b)/(abs(a)+abs(b)) if (abs(a)+abs(b))!=0 else 0 for a,b in zip(pointA, pointB)[:-1] )
	print 'hello canberra'

def macro_f_score_n_accuracy(predicted_label, actual_label):
	'''set table to size[10][10]'''
	table = [[0 for i in range(10)] for j in range(10)]
	total = len(predicted_label)

	'''record in table'''
	for i in range(len(predicted_label)):
		table[int(actual_label[i])-1][int(predicted_label[i])-1] += 1

	'''get f1 score and acc for each categories'''
	f1, acc = [], []
	for i in range(10):
		TP = table[i][i]
		FP = (-1)*table[i][i]
		for j in range(10):	FP += table[j][i]
		FN = (-1)*table[i][i]
		for j in range(10):	FN += table[i][j]
		TN = total-TP-FP-FN

		acc.append((TP+TN)/float(total))
		if float(2*TP + FN + FP) == 0:	f1.append(0)
		else: f1.append(2*TP/float(2*TP + FN + FP))
	macro_f1 = sum(f1)/10
	macro_acc = sum(acc)/10
	# print ' F1 Score: '+str(macro_f1),
	# print '\t Accuracy: '+str(macro_acc)
	return [macro_f1, macro_acc]

def majority(arr, k):
	# print '  (get the majority)'
	dic = {}
	# print arr
	for i in range(k):
		dic[arr[i][1]] = 1 if arr[i][1] not in dic else dic[arr[i][1]]+1
	value, key = max((value, key) for key, value in dic.items())
	return key
def knn_classify(train_set, test_set, i, k):
	actual_label = []
	predicted_label = []
	for test in test_set:
		distance = []
		for train in train_set:
			distance.append([distance_measures(train, test, i), train[-1]])
		sort_dist = sorted(distance, key=lambda neigh: neigh[0])
		predicted_label.append(majority(sort_dist, k))
		actual_label.append(test[-1])
	return macro_f_score_n_accuracy(predicted_label, actual_label)

def split_into_n(n, data):
	print ' >  split into n fold(s)'
	df = [[] for i in range(n)]
	for i, line in enumerate(data):
		df[i%n].append(line)
	return df
def normalize(data):
	print ' >  normalization'
	min_col = np.min(data, axis=0)
	max_col = np.max(data, axis=0)
	normal_data = []
	for line in data:
		row = []
		'''label doesn't need to be normalize'''
		for i, col in enumerate(line[:-1]):
			row.append((float(col)-min_col[i]) / (max_col[i]-min_col[i]))
		'''set the label type to int'''
		row.append(int(line[-1]))
		normal_data.append(np.array(row))
	return normal_data
def read_data(file_name):
	print ' >  read data'
	with open(file_name, 'rb') as csvfile:
		content_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
		data_n_label = []
		for i,row in enumerate(content_reader):
			if i==0:	
				column_name = row
				print column_name
				continue
			data_n_label.append(map(float, row))
	return np.array(data_n_label)

def main():
	'''n validation'''
	n = 4

	'''data preprocess'''
	data_n_label = read_data('winequality-white.csv')
	# data_n_label = read_data('tmp_wine.csv')
	print len(data_n_label)
	np.random.shuffle(data_n_label)
	normal_data = normalize(data_n_label)
	df = split_into_n(n, normal_data)
	# print df
	print 'n = '+str(len(df))
	print 'each fold has '+str(len(df[0]))

	'''tune the hyperparameter'''
	for i in range(3):		# diff distanct metrics
		for k in range(1,10,2):
			if i==0:	print ' [#####] euclidean',
			elif i==1:	print ' [#####] manhattan',
			elif i==2:	print ' [#####] canberra',
			else:		print ' [#####] wait whaaaaaat'
			print ' with k='+str(k),

			'''4-fold cross validation'''
			valid_score = []
			test_score  = []
			'''tune the hyperparameter'''
			for j in range(n):
				# print '\n  [#]   fold'+str(i)+' as test set'
				test_set = df.pop(j)
				train_n_valid_set = np.vstack(df)
				df.insert(j, test_set)
				
				valid_n = len(train_n_valid_set)//5	# 20% as validation set
				validation_set = train_n_valid_set[:valid_n]
				training_set = train_n_valid_set[valid_n:]

				# print '  [#]   Validation: ',
				valid_score.append(knn_classify(training_set, validation_set, i, k))
				# print '  [#]   Test:       ',
				test_score.append(knn_classify(training_set, test_set, i, k))

			# print '\n  [#]   Average: '
			print '\n  [#]   Validation: ',
			tmp_sum = 0
			for t in range(n):	tmp_sum += valid_score[t][0]
			print ' F1 Score: '+str(tmp_sum/4),
			tmp_sum = 0
			for t in range(n):	tmp_sum += valid_score[t][1]
			print '\n\t\t\t\t\t  Accuracy: '+str(tmp_sum/4)
			print '  [#]   Test:       ',
			tmp_sum = 0
			for t in range(n):	tmp_sum += test_score[t][0]
			print ' F1 Score: '+str(tmp_sum/4),
			tmp_sum = 0
			for t in range(n):	tmp_sum += test_score[t][1]
			print '\n\t\t\t\t\t  Accuracy: '+str(tmp_sum/4)
			print ''
			


if __name__ == '__main__':

	main()

