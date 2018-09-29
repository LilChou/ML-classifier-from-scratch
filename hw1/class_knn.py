#! /usr/bin/env python

# CS578 Hw1 Fall18 @ Purdue University
# @author Yu-Jung Chou
# @data Sep 19 2018

import numpy as np
import csv
from heapq import *
# import matplotlib.pyplot as plt
import time

def print_score(head, acc, f1, p, r):
	print head,
	print 'accuracy: '+'{0:.3f}'.format(acc)+'%\tmacro_f1: '+'{0:.3f}'.format(f1)
	# print '%\tmacro_precision: '+'{0:.3f}'.format(p)+'%  \tmacro_recall: '+'{0:.3f}'.format(r)+'%'	
def f_write_score(head, f, acc, f1, p, r):
	f.write(head)
	f.write('accuracy: '+'{0:.3f}'.format(acc)+'%\tmacro_f1: '+'{0:.3f}'.format(f1),)
	f.write('%\tmacro_precision: '+'{0:.3f}'.format(p)+'%  \tmacro_recall: '+'{0:.3f}'.format(r)+'%')
			
def accuracy(predict_label, true_label):
	# two input should be np.array
	return np.mean(predict_label==true_label)*100
def macro_f1_score(predict_label, true_label):
	table = [[0 for i in range(10)]for i in range(10)]
	total = len(predict_label)
	for i in range(total):
		table[int(true_label[i])][int(predict_label[i])] += 1
	# print np.array(matrix)

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
def scoring(predict_label, true_label):
	acc = accuracy(predict_label, true_label)
	f1, p, r = macro_f1_score(predict_label, true_label)
	# print 'accuracy: '+'{0:.3f}'.format(acc)+'%\tmacro_f1: '+'{0:.3f}'.format(f1),
	# print '%\tmacro_precision: '+'{0:.3f}'.format(p)+'%  \tmacro_recall: '+'{0:.3f}'.format(r)+'%'
	return [acc, f1, p, r]

def distance_measures(pointA, pointB, index_of_measures):
	if index_of_measures==0:	return euclidean_distance(pointA, pointB)
	if index_of_measures==1:	return manhattan_distance(pointA, pointB)
	if index_of_measures==2:	return canberra_distance(pointA, pointB)
def euclidean_distance(pointA, pointB):
	# last col is label
	# return sqrt(sum( (a-b)**2 for a,b in zip(pointA, pointB)[:-1] ))
	return np.sqrt(np.sum( np.square(a-b) for a,b in zip(pointA, pointB)[:-1] ))
	print 'hello euclidean'
def manhattan_distance(pointA, pointB):
	# last col is label
	# return sum( abs(a-b) for a,b in zip(pointA, pointB)[:-1] )
	return np.sum( np.absolute(a-b) for a,b in zip(pointA, pointB)[:-1] )
	print 'hello manhattan'
def canberra_distance(pointA, pointB):
	# last col is label
	# return sum( abs(a-b)/(abs(a)+abs(b)) if (abs(a)+abs(b))!=0 else 0 for a,b in zip(pointA, pointB)[:-1] )
	return np.sum( np.absolute(a-b)/(np.absolute(a)+np.absolute(b)) if a!=0 and b!=0 else 0 for a,b in zip(pointA, pointB)[:-1])
	print 'hello canberra'

class KNN:
	distance_dic = {
		'euclidean':0,
		'manhattan':1,
		'canberra':2,
	}
	def __init__(self, distance_func, k):
		# self.dist = distance_func
		self.dist = self.distance_dic[distance_func]
		self.k = k
		self.trained = 0

	def train(self, data):
		self.trained = 1
		self.neigh = data

		# # normalization
		# self.neigh = normalize(self.neigh)

	def majority(self, distance, k):
		dic = {}
		for i in range(k):
			cur = heappop(distance)
			dic[cur[1]] = 1 if cur[1] not in dic else dic[cur[1]]+1
		value, key = max((value, key) for key, value in dic.items())
		return key
	def classify(self, data):
		prediction = []
		if self.trained==0:	
			print '\n\n >  PLEASE TRAIN BEFOR YOU CLASSIFY!!!!\n\n'
			return []

		for node in data:
			# default heap is min heap
			distance_min_heap = []
			for n in self.neigh:
				heappush(distance_min_heap, [distance_measures(n, node, self.dist), n[-1]])
			# print distance_min_heap
			prediction.append(self.majority(distance_min_heap, self.k))
		return np.array(prediction)

def split_fold(data, n):
	sets = [[]for i in range(n)]
	for i, row in enumerate(data):
		sets[i%n].append(row)
	return np.array(sets)
def split_data(data, n):
	num = len(data)//n
	return [data[:num], data[num:]]
def normalize(array_2d):
	col_max = np.max(array_2d, axis=0)
	col_min = np.min(array_2d, axis=0)

	for row in range(len(array_2d)):
		# last col is label, no need to normalize
		for col in range(len(array_2d[0])-1):
			array_2d[row][col] = (float(array_2d[row][col])-col_min[col])/(col_max[col]-col_min[col])
	return array_2d
def read_data(file_name):
	# print ' >  read data'
	with open(file_name, 'r') as csvfile:
		content_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
		col = []
		data = []
		for i, row in enumerate(content_reader):
			if i==0:
				col = row
			else:
				data.append(np.array(row, dtype='f'))
	return col, data
def tune():
	start = time.time()
	# hyperparameters
	distance_funcs = ['euclidean', 'manhattan', 'canberra']
	ks = [1,3,5,7,9]
	n = 4	# num of folds

	# read in data and data preprocessing
	# col, data = read_data('tmp_wine.csv')
	col, data = read_data('winequality-white.csv')
	
	np.random.shuffle(data)
	norm_data = normalize(data)
	
	# 4 folds
	sets = split_fold(norm_data, n)
	
	# a file to record output for tuning
	f = open('tune_knn.txt', 'w+')

	# start tuning
	# for d_ind, distance_func in enumerate(distance_funcs):
	distance_func = 'canberra'
	valid_acc_plt = []
	valid_f1_plt = []
	for k in ks:
		f.write('\n----------------------------------------')
		# print ('\n----------------------------------------')
		
		valid_score = []
		test_score = []
		# start each fold
		for i in range(n):
			# model initialization
			m = KNN(distance_func, k)

			# test and train
			test = sets[i]
			tmp_l = [sets[j] if j!= i else [] for j in range(n)]
			tmp_l.pop(i)
			train = np.vstack(tmp_l)

			# 20% as validation data
			valid, train = split_data(train, 5)

			m.train(train)

			print 'fold '+str(i)+' where distance function = '+distance_func+' and k='+str(k)
			
			print ' >  valid set\t',
			true_label = np.array([row[-1] for row in valid])
			predict_label = m.classify(valid)
			valid_score.append(scoring(predict_label, true_label))

			print ' >  test set \t',
			true_label = np.array([row[-1] for row in test])
			predict_label = m.classify(test)
			test_score.append(scoring(predict_label, true_label))			


		f.write('\nAVERAGE distance function: '+distance_func+'\tk='+str(m.k))
		print 'AVERAGE distance function: '+distance_func+'\tk='+str(m.k)

		acc, f1, p, r = np.average(valid_score, axis=0)
		f_write_score('\n >  \tvalid set\t', f, acc, f1, p, r)
		print_score(' >  \tvalid set\t', acc, f1, p, r)
		valid_acc_plt.append(acc)
		valid_f1_plt.append(f1)

		acc, f1, p, r = np.average(test_score, axis=0)
		f_write_score('\n >  \ttest set\t', f, acc, f1, p, r)
		print_score(' >  \ttest set\t', acc, f1, p, r)
			
	
	f.write("\n--- %s seconds ---\n" % (time.time()-start))
	# print "--- %s seconds ---" % (time.time()-start)
	# plt.figure(d_ind+1)
	# plt.plot(ks, valid_acc_plt, color='blue', linestyle='-', label='accuracy')
	# plt.plot(ks, valid_f1_plt, color='red', linestyle='-', label='f1 score')
	# plt.title('KNN - Euclidean Distance')
	# plt.xlabel('K (k nearest neighbors)')
	# plt.ylabel('%')
	# plt.grid(True)

	# plt.show()
	
	f.close()
	print 'end of the program, output in tune_knn.txt'			
def final_hyperparameters(distance_func, k):
	# start = time.time()
	# hyperparameters
	n = 4	# num of folds

	# read in data and data preprocessing
	col, data = read_data('winequality-white.csv')
	
	np.random.shuffle(data)
	norm_data = normalize(data)
	
	# 4 folds
	sets = split_fold(norm_data, n)
	
	# a file to record output for tuning
	# f = open('tune_knn.txt', 'w+')

	# start tuning
	# for d_ind, distance_func in enumerate(distance_funcs):
	distance_func = 'euclidean'
	valid_acc_plt = []
	valid_f1_plt = []
	# f.write('\n----------------------------------------')
	# print ('\n----------------------------------------')
	
	valid_score = []
	test_score = []
	# start each fold
	for i in range(n):
		# model initialization
		m = KNN(distance_func, k)

		# test and train
		test = sets[i]
		tmp_l = [sets[j] if j!= i else [] for j in range(n)]
		tmp_l.pop(i)
		train = np.vstack(tmp_l)

		# 20% as validation data
		valid, train = split_data(train, 5)

		m.train(train)

		print ' >  fold '+str(i)# +' where distance function = '+distance_func+' and k='+str(k)
		
		true_label = np.array([row[-1] for row in valid])
		predict_label = m.classify(valid)
		acc, f1, p, r = scoring(predict_label, true_label)
		print_score(' >  \tvalid set\t', acc, f1, p, r)
		valid_score.append([acc, f1, p, r])
		# valid_score.append(scoring(predict_label, true_label))

		true_label = np.array([row[-1] for row in test])
		predict_label = m.classify(test)
		acc, f1, p, r = scoring(predict_label, true_label)
		print_score(' >  \ttest set \t', acc, f1, p, r)
		test_score.append([acc, f1, p, r])
		# test_score.append(scoring(predict_label, true_label))			


	# f.write('\nAVERAGE distance function: '+distance_func+'\tk='+str(m.k))
	print ' >  AVERAGE'# distance function: '+distance_func+'\tk='+str(m.k)

	acc, f1, p, r = np.average(valid_score, axis=0)
	# f_write_score('\n >  \tvalid set\t', f, acc, f1, p, r)
	print_score(' >  \tvalid set\t', acc, f1, p, r)


	acc, f1, p, r = np.average(test_score, axis=0)
	# f_write_score('\n >  \ttest set\t', f, acc, f1, p, r)
	print_score(' >  \ttest set\t', acc, f1, p, r)
			
	
	# f.write("\n--- %s seconds ---\n" % (time.time()-start))
	# f.close()
if __name__=='__main__':
	# tune()

	k = 1
	distance_func = 'manhattan'
	print 'Hyper-parameters:'
	print 'K: '+str(k)
	print 'Distance measure: '+distance_func
	final_hyperparameters(distance_func, k)