import networkx as nx
from networkx.readwrite import json_graph
import json
import pickle
import sys
import time
import snap

entity_to_idx = json.load(open("entity_to_idx_dbpedia_pagelink.json"))
num_nodes = len(list(entity_to_idx.keys()))
del(entity_to_idx)

with open("dbpedia_pagelink_edge_tuple_list.pkl", "rb") as fp1:
    edge_tuple_list = pickle.load(fp1)

print('Begin snap graph creation')
G = snap.TUNGraph.New()

for i in range(num_nodes):
    G.AddNode(i)
print('Added nodes')

count=0
for item in edge_tuple_list:
    G.AddEdge(item[0], item[1])
    count = count+1
    if count%100000 == 0:
        print('Done with '+str(count))
del(edge_tuple_list)
print('Added edges')

print('Is connected: ')
print(snap.IsConnected(G))

print('Is weakly connected: ')
print(snap.IsWeaklyConn(G))

print('Fraction of nodes in maximum weakly connected component: ')
print(snap.GetMxWccSz(G))

idx_to_entity = json.load(open("idx_to_entity_dbpedia_pagelink.json"))

fp2 = open('dbpedia_pagelink_entity_pair_list_max_connected_component.txt', 'w')

MxWcc = snap.GetMxWcc(G)
for EI in MxWcc.Edges():
    fp2.write(idx_to_entity[str(EI.GetSrcNId())]+' '+idx_to_entity[str(EI.GetDstNId())]+'\n')

fp2.close()

del(idx_to_entity)

print('Done')




