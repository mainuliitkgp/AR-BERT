import sys
import numpy as np
import math
import pickle

unique_entity_fp = sys.argv[1]
concatenated_embedding_fp = sys.argv[2]
ds_name = sys.argv[3]

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# prepare unique entity list
unique_entity_list = []
with open(unique_entity_fp) as fp:
    for line in fp:
        unique_entity_list.append(line.strip())

# read concatenated entity embedding
concatenated_embedding = np.load(concatenated_embedding_fp)

# list of sigmoid of dot products of two entity embedding
sigmoid_tuple_list = []
for i in range(len(unique_entity_list)):
    for j in range(i+1, len(unique_entity_list)):
        sigmoid_tuple_list.append((i, j, sigmoid(np.dot(concatenated_embedding[i], concatenated_embedding[j]))))

with open(ds_name+'_sigmoid_dot_product_tuple_list.pkl', 'wb') as fp:
    pickle.dump(sigmoid_tuple_list, fp)

# sort nodes wrt. sigmoid of dot product for rach node
adjacency_sigmoid_list = {} # {node1:[(node2, sigmoid_val)]}
for tuple_element in sigmoid_tuple_list:
    node1 = int(tuple_element[0])
    node2 = int(tuple_element[1])
    sigmoid_val = tuple_element[2]
    if node1 not in adjacency_sigmoid_list:
        adjacency_sigmoid_list[node1] = []
    if node2 not in adjacency_sigmoid_list:
        adjacency_sigmoid_list[node2] = []

    adjacency_sigmoid_list[node1].append((node2, sigmoid_val))
    adjacency_sigmoid_list[node2].append((node1, sigmoid_val))

for i in range(len(unique_entity_list)):
    adjacency_sigmoid_list[i].sort(reverse = True, key=lambda x: x[1])

with open(ds_name+'_adjacency_sigmoid_list.json', 'w') as fp:
    json.dump(adjacency_sigmoid_list, fp)

all_node_sorted_node_list_wrt_sigmoid_dot = []
for i in range(len(unique_entity_list)):
    sorted_node_list_wrt_sigmoid_dot = []
    for tuple_item in adjacency_sigmoid_list[i]:
        sorted_node_list_wrt_sigmoid_dot.append(tuple_item[0])
    all_node_sorted_node_list_wrt_sigmoid_dot.append(sorted_node_list_wrt_sigmoid_dot)

with open(ds_name+'_soreted_node_list_wrt_sigmoid_dot.pkl', 'wb') as fp:
    pickle.dump(all_node_sorted_node_list_wrt_sigmoid_dot, fp) 


 

 
