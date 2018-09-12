#! /usr/bin/env python
import sys
def bfs(map_edge, map_node, record, bfs_list, ans):
	if bfs_list == []:	return ans
	
	new_bfs = set()
	for node in bfs_list:
		if node in record:	continue
		record.append(node)
		if ans[1]>map_node[node]:	ans = [node, map_node[node]]
		for linked in map_edge[node]:
			if linked not in record:
				new_bfs.add(linked)

	ans = bfs(map_edge, map_node, record, list(new_bfs), ans)
	return ans

def main():
	if len(sys.argv) == 1:
		print "[X] Wrong usage!!"
		print "[*] Usage: ./hw0 data_file"
		exit(0)
	file = sys.argv[1]
	f = open(file, 'r')
	map_edge = {}
	map_node = {}
	ans = [float('inf'), float('inf')]

	# read in data, and store them into _edge & _node
	while f:
		tmp = f.readline()
		if tmp == '':	break
		g = 2
		tt = ''
		id_1, id_2 = 0, 0
		val_1, val_2 = 0, 0
		for i in range(1,30):
			if tmp[i].isdigit():
				tt += tmp[i]
			elif tmp[i]==':':
				id_2 = int(tt)
				tt = ''
			elif tmp[i]=='>':	
				g -= 1
				if g==1:
					val_1 = int(tt)
					id_1 = id_2
					tt = ''
				elif g==0:
					val_2 = int(tt)
					break
		# print str(id_1)+'...'+str(val_1)
		# print str(id_2)+'...'+str(val_2)

		if id_1 not in map_node:	
			map_node[id_1] = val_1
			map_edge[id_1] = [id_2]
		else:	map_edge[id_1].append(id_2)
		if id_2 not in map_node:	
			map_node[id_2] = val_2
			map_edge[id_2] = [id_1]
		else:	map_edge[id_2].append(id_1)

		# break

	# initialize the bfs_list
	bfs_list = [map_edge.keys()[0]]
	
	print '[*] Please be patient ...'

	# start bfs	
	ans = bfs(map_edge, map_node, [], bfs_list, ans)

	print '[O] ans[NodeId, NodeValue] =',
	print ans






























if __name__ == "__main__":
	main()
