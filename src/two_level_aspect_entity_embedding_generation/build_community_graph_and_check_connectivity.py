import json
from itertools import combinations, permutations
import pickle
import time
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

# read louvain cluster dictionary {cluster_id:node_list} and {node:cluster_id} dictionary
with open("cluster_dict.pkl", "rb") as fp1:
    cluster_dict = pickle.load(fp1)

with open("node_to_cluster_id_dict.pkl", "rb") as fp2:
    node_to_cluster_id_dict = pickle.load(fp2)

# build community graph

## creating edge tuple list
with open("connected_dbpedia_pagelink_edge_tuple_list.pkl", "rb") as fp3:
    edge_tuple_list = pickle.load(fp3)

cluster_edge_count_dict = {}

for line in edge_tuple_list:
    id1 = node_to_cluster_id_dict[int(line[0])]
    id2 = node_to_cluster_id_dict[int(line[1])]
    if id1 != id2:
        if (id1, id2) not in cluster_edge_count_dict:
            cluster_edge_count_dict[(id1, id2)] = 0
        cluster_edge_count_dict[(id1, id2)] = cluster_edge_count_dict[(id1, id2)] + 1

community_graph_edge_list = []

for id1, id2 in list(combinations(list(cluster_dict.keys()), 2)):
    edge_count = 0
    if (id1, id2) in cluster_edge_count_dict:
        edge_count = edge_count + cluster_edge_count_dict[(id1, id2)]

    if (id2, id1) in cluster_edge_count_dict:
        edge_count = edge_count + cluster_edge_count_dict[(id2, id1)] 

    all_possible_edge_count = len(cluster_dict[id1]) * len(cluster_dict[id2])

    edge_weight = edge_count/all_possible_edge_count
    
    community_graph_edge_list.append((id1, id2, edge_weight))

## creating nodes
nodes=[]
node={}
for i in range(len(list(cluster_dict.keys()))):
    node['test']=False
    node['id']=i
    node['val']=False
    nodes.append(node)
    node={}

## creating edges
weighted_adjacency_matrix = np.zeros((len(list(cluster_dict.keys())), len(list(cluster_dict.keys()))))
edges=[]
edge={}
for item in community_graph_edge_list:
    if item[2] > 0:
        weighted_adjacency_matrix[item[0]][item[1]] = item[2]
        weighted_adjacency_matrix[item[1]][item[0]] = item[2]
        edge['source']=min(item[0], item[1])
        edge['target']=max(item[0], item[1])
        edges.append(edge)
        edge={}

## creating graph
G={}
G['directed']=False
G['graph']={}
G['nodes']=nodes
G['links']=edges
G['multigraph']=False

with open('community_graph_dbpedia_pagelink-G.json', 'w') as fp:
    json.dump(G, fp)

## creating node_id to id integer
node_id_dict={}
for i in range(len(list(cluster_dict.keys()))):
    node_id_dict[str(i)]=i
with open('community_graph_dbpedia_pagelink-id_map.json', 'w') as fp:
    json.dump(node_id_dict, fp)

## saving weighted adjacency matrix
np.save('community_graph_dbpedia_pagelink-weighted_adjacency_matrix.npy',weighted_adjacency_matrix)

G = json.load(open("community_graph_dbpedia_pagelink-G.json"))
G = json_graph.node_link_graph(G)

print(nx.is_connected(G))
