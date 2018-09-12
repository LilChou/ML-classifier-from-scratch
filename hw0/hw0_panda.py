#! /usr/bin/env python
import pandas as pd

def main():
	print "helloworld"

	# # read csv file
	df = pd.read_csv('nodes.csv', header=None)
	# # print df.shape

	ans_id = df.loc[0][0][1:-1].split(':')[0]
	ans_value = df.loc[0][0][1:-1].split(':')[1]
	ans = [ans_id, ans_value]
	# print ans

	bfs_list = [df.loc[0][0]]
	record = []

	def bfs(bfs_list, ans, record):
		# print 'ans='+str(ans)
		next_list = set()
		for node in bfs_list:
			if node in record:	continue
			record.append(node)

			if ans[1] > int(node[1:-1].split(':')[1]):
				ans[0] = int(node[1:-1].split(':')[0])
				ans[1] = int(node[1:-1].split(':')[1])
				# print 'ans='+str(ans)

			for tmp in df.loc[df[0]==node][1]:
				next_list.add(tmp[1:])

		if len(next_list) >0:
			bfs(list(next_list), ans, record)

	bfs(bfs_list, ans, record)

	print 'Node with smallest value:',
	print ans







if __name__=='__main__':
	main()


	