#! /usr/bin/env python

# CS578 Hw1	Fall18 @ Purdue University
# @author Yu-Jung Chou
# @date	Sep. 19 2018

import numpy as np
import csv
from heapq import *
import matplotlib.pyplot as plt
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

def entropy(data):
	# label is the last col of data
	dic = {}
	l = len(data)
	for row in data:
		dic[row[-1]] = 1 if row[-1] not in dic else dic[row[-1]]+1

	return (-1)*np.sum([ dic[item]/float(l)*np.log2(dic[item]/float(l)) for item in dic ])
def divide_by_threshold(data, col, threshold):
	true, false = [], []
	for row in data:
		if row[col] >= threshold:	true.append(row)
		else:						false.append(row)
	return [true, false]

class TreeNode():
	def __init__(self, column_asked, threshold, dept):
		self.col = column_asked
		self.t = threshold
		self.d = dept
		self.true = None
		self.false = None
		self.is_leaf = False
class LeafNode():
	def __init__(self, major_label):
		self.label = major_label
		self.is_leaf = True
class DecisionTree:
	"""docstring for Decision Tree"""
	def __init__(self):
		# print ' >  A new tree'
		self.head = None
	def grow_tree(self, data, d):
		if entropy(data)==0:					return LeafNode(data[0][-1])
		total = len(data)
		new_entropy_min_heap = []
		# search through all cols, exclude col label
		for i in range(len(data[0])-1):
			threshold = set([data[j][i] for j in range(len(data))])
			for t in threshold:
				true, false = divide_by_threshold(data, i, t)
				cur_entropy = (len(true)*entropy(true) + len(false)*entropy(false))/float(total)
				heappush(new_entropy_min_heap, [cur_entropy, i, t])
		# get the best question with best info gain (min entropy)
		best_q = heappop(new_entropy_min_heap)

		# construct the new treenode
		cur_node = TreeNode(best_q[1], best_q[2], d+1)

		# prepare to grow true branch and false branch
		true, false = divide_by_threshold(data, cur_node.col, cur_node.t)
		cur_node.true = self.grow_tree(true, d+1)
		cur_node.false = self.grow_tree(false, d+1)
		return cur_node
	def train(self, data):
		self.head = self.grow_tree(data, 0)

	def majority(self, data):
		dic = {}
		for row in data:
			dic[row[-1]] = 1 if row[-1] not in dic else dic[row[-1]]+1
		value, key = max((value, key) for key, value in dic.items())
		return key
	def grow_tree_with_hp(self, data, d, max_d):
		if entropy(data)==0:					return LeafNode(data[0][-1])
		if d+1 > max_d:							return LeafNode(self.majority(data))
		
		total = len(data)
		new_entropy_min_heap = []
		# search through all cols, exclude col label
		for i in range(len(data[0])-1):
			threshold = set([data[j][i] for j in range(len(data))])
			for t in threshold:
				true, false = divide_by_threshold(data, i, t)
				cur_entropy = (len(true)*entropy(true) + len(false)*entropy(false))/float(total)
				heappush(new_entropy_min_heap, [cur_entropy, i, t])
		# get the best question with best info gain (min entropy)
		best_q = heappop(new_entropy_min_heap)

		# construct the new treenode
		cur_node = TreeNode(best_q[1], best_q[2], d+1)

		# prepare to grow true branch and false branch
		true, false = divide_by_threshold(data, cur_node.col, cur_node.t)
		cur_node.true = self.grow_tree_with_hp(true, d+1, max_d)
		cur_node.false = self.grow_tree_with_hp(false, d+1, max_d)
		return cur_node
	def train_with_hp(self, data, d):
		self.head = self.grow_tree_with_hp(data, 0, d)

	def print_tree(self):
		if self.head==None:		print '\n\n >  PLEASE TRAIN TO GET THE TREE!!!!\n\n'
		print 'print out the tree nodes, format: col | threshold'
		ll = [self.head]
		while ll:
			new_l = []
			for node in ll:
				if node.is_leaf:
					print ('leaf node of label: %d' % (node.label))
					continue
				print ('col: %2d | t: %f' % (node.col, node.t))
				if node.true:	new_l.append(node.true)
				if node.false:	new_l.append(node.false)
			ll = new_l
		print 'end of print tree'
	def classify(self, data):
		if self.head==None:	
			print '\n\n >  PLEASE TRAIN BEFOR YOU CLASSIFY!!!!\n\n'
			return []

		prediction = []
		for row in data:
			cur_node = self.head
			while not cur_node.is_leaf:
				if row[cur_node.col] >= cur_node.t:	cur_node = cur_node.true
				else:								cur_node = cur_node.false
			prediction.append(cur_node.label)
		return np.array(prediction)


def split_fold(data, n):
	sets = [[]for i in range(n)]
	for i, row in enumerate(data):
		sets[i%n].append(row)
	return np.array(sets)
def split_data(data, n):
	num = len(data)//n
	return [data[:num], data[num:]]
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
	dept_limit = [i for i in range(21)]					# max dept of tree
	# min_sample_split = [10, 20, 30, 50, 100]			# min samples to be split
	n = 4 												# num of folds
	
	# read in data and data preprocessing
	# col, data = read_data('tmp_wine.csv')
	col, data = read_data('winequality-white.csv')

	np.random.shuffle(data)

	# 4 folds
	sets = split_fold(data, n)

	# a file to record output for tuning
	f = open('tune_dt.txt', 'w+')

	valid_acc_plt = []
	valid_f1_plt = []

	# start tuning
	for dept in dept_limit:
		f.write('\n----------------------------------------\n')

		train_score = []
		valid_score = []
		test_score = []
		# start each fold
		for i in range(n):
			# model initialization
			m = DecisionTree()

			# test and train
			test = sets[i]
			tmp_l = [sets[j] if j!=i else [] for j in range(n)]
			tmp_l.pop(i)
			train = np.vstack(tmp_l)

			# 20% as validation data
			valid, train = split_data(train, 5)

			# m.train(train)
			m.train_with_hp(train, dept)	# hp:	hyperparameters
			# m.print_tree()

			print('fold '+str(i)+' where max_dept='+str(dept))
			print ' >  train set\t',
			true_label = np.array([row[-1] for row in train])
			predict_label = m.classify(train)
			train_score.append(scoring(predict_label, true_label))
			
			print ' >  valid set\t',
			true_label = np.array([row[-1] for row in valid])
			predict_label = m.classify(valid)
			valid_score.append(scoring(predict_label, true_label))
			
			print ' >  test set \t',
			true_label = np.array([row[-1] for row in test])
			predict_label = m.classify(test)
			test_score.append(scoring(predict_label, true_label))

		f.write('\nAVERAGE where max_dept='+str(dept))
		print 'AVERAGE where max_dept='+str(dept)

		acc, f1, p, r = np.average(train_score, axis=0)
		f_write_score('\n >  \ttrain set\t', f, acc, f1, p, r)
		print_score(' >  \ttrain set\t', acc, f1, p, r)
		# train_plt.append(acc)

		acc, f1, p, r = np.average(valid_score, axis=0)
		f_write_score('\n >  \tvalid set\t', f, acc, f1, p, r)
		print_score(' >  \tvalid set\t', acc, f1, p, r)
		valid_acc_plt.append(acc)
		valid_f1_plt.append(f1)
		
		acc, f1, p, r = np.average(test_score, axis=0)
		f_write_score('\n >  \ttest set\t', f, acc, f1, p, r)
		print_score(' >  \ttest set\t',acc, f1, p, r)
		# test_plt.append(acc)


	f.write('\n')
	f.close()
	print "--- %s seconds ---" % (time.time()-start)
	plt.plot(dept_limit, valid_acc_plt, 'b--', label='accuracy')
	plt.plot(dept_limit, valid_f1_plt, 'r--', label='f1 score')
	# plt.plot(dept_limit, train_plt, 'ro', dept_limit, valid_plt, 'b--', dept_limit, test_plt, 'g*')
	plt.title('Decision Tree')
	plt.xlabel('d (max depth of the tree)')
	plt.ylabel('(%)')
	plt.legend()
	plt.grid(True)
	plt.show()
	print 'end of the program, output in the tune_dt.txt'
def final_hyperparameters(dept):
	start = time.time()
	# hyperparameters
	n = 4 		# num of folds
	
	# read in data and data preprocessing
	# col, data = read_data('tmp_wine.csv')
	col, data = read_data('winequality-white.csv')

	np.random.shuffle(data)

	# 4 folds
	sets = split_fold(data, n)

	# # a file to record output for tuning
	# f = open('tune_dt.txt', 'w+')

	train_score = []
	valid_score = []
	test_score = []

	# start each fold
	for i in range(n):
		# model initialization
		m = DecisionTree()

		# test and train
		test = sets[i]
		tmp_l = [sets[j] if j!=i else [] for j in range(n)]
		tmp_l.pop(i)
		train = np.vstack(tmp_l)

		# 20% as validation data
		valid, train = split_data(train, 5)

		# m.train(train)
		m.train_with_hp(train, dept)
		# m.print_tree()

		print(' >  fold '+str(i))#+' where max_dept='+str(dept))
		
		true_label = np.array([row[-1] for row in train])
		predict_label = m.classify(train)
		acc, f1, p, r = scoring(predict_label, true_label)
		print_score(' >  \ttrain set\t', acc, f1, p, r)
		train_score.append([acc, f1, p, r])
		# train_score.append(scoring(predict_label, true_label))
		
		
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

	# f.write('\nAVERAGE where max_dept='+str(dept))
	print ' >  AVERAGE'

	acc, f1, p, r = np.average(train_score, axis=0)
	# f_write_score('\n >  \ttrain set\t', f, acc, f1, p, r)
	print_score(' >  \ttrain set\t', acc, f1, p, r)
	# train_plt.append(acc)

	acc, f1, p, r = np.average(valid_score, axis=0)
	# f_write_score('\n >  \tvalid set\t', f, acc, f1, p, r)
	print_score(' >  \tvalid set\t', acc, f1, p, r)
	
	acc, f1, p, r = np.average(test_score, axis=0)
	# f_write_score('\n >  \ttest set\t', f, acc, f1, p, r)
	print_score(' >  \ttest set\t',acc, f1, p, r)
	# test_plt.append(acc)


	# f.write('\n')
	# f.close()
	print "--- %s seconds ---" % (time.time()-start)
if __name__=='__main__':
	# tune()

	dept = 15
	print 'Hyper-parameters:'
	print 'Max-Dept: '+str(dept)
	final_hyperparameters(dept)