import sys
import numpy as np
import json
import pickle

unique_entity_fp = sys.argv[1]
sub_graph_embedding_fp = sys.argv[2]
sub_graph_embedding_indices_fp = sys.argv[3]
connected_entity_to_idx_dict_fp = sys.argv[4]
node_to_cluster_id_dict_fp = sys.argv[5]
cluster_kg_embedding_fp = sys.argv[6]
cluster_kg_embedding_indices_fp = sys.argv[7]
ds_name = sys.argv[8]

# prepare unique entity list
unique_entity_list = []
with open(unique_entity_fp) as fp:
    for line in fp:
        unique_entity_list.append(line.strip())

# read sub-graph entity embedding and corresponding indices
sub_graph_entity_embedding = np.load(sub_graph_embedding_fp)

sub_graph_entity_embedding_indices = []
with open(sub_graph_embedding_indices_fp) as fp:
    for line in fp:
        sub_graph_entity_embedding_indices.append(int(line.strip()))

# re-arrange sub-graph entity embedding wrt. unique entities
rearranged_sub_graph_entity_embedding = []
for i, entity in enumerate(unique_entity_list):
    idx = sub_graph_entity_embedding_indices.index(i)
    emd = sub_graph_entity_embedding[idx]
    rearranged_sub_graph_entity_embedding.append(emd)

rearranged_sub_graph_entity_embedding = np.array(rearranged_sub_graph_entity_embedding, dtype = np.float32) 

# read connected KG entity to id dictionary and map sub-graph entity index to connected KG entity index
connected_entity_to_idx_dict = json.load(open(connected_entity_to_idx_dict_fp))

sub_graph_entity_idx_to_connected_kg_idx = {}
for i, entity in enumerate(unique_entity_list):
    try:
        sub_graph_entity_idx_to_connected_kg_idx[int(i)] = connected_entity_to_idx_dict[entity.strip()]
    except:
        sub_graph_entity_idx_to_connected_kg_idx[int(i)] = -1 # for not mapped sub-graph entities in connected KG

# map sub-graph entity index to cluster id
with open(node_to_cluster_id_dict_fp, "rb") as fp:
    node_to_cluster_id_dict = pickle.load(fp)

sub_graph_entity_idx_to_cluster_id_dict = {}
for i, entity in enumerate(unique_entity_list):
    node_id_in_kg = sub_graph_entity_idx_to_connected_kg_idx[int(i)]
    if node_id_in_kg != -1:
        sub_graph_entity_idx_to_cluster_id_dict[int(i)] = node_to_cluster_id_dict[int(node_id_in_kg)]
    else:
        sub_graph_entity_idx_to_cluster_id_dict[int(i)] = -1 # no cluster membership  

# read cluster KG embedding and corresponding indices
cluster_kg_embedding = np.load(cluster_kg_embedding_fp)

cluster_kg_embedding_indices = []
with open(cluster_kg_embedding_indices_fp) as fp:
    for line in fp:
        cluster_kg_embedding_indices.append(int(line.strip()))

# re-arrange cluster KG embedding wrt. unique entities
rearranged_cluster_kg_entity_embedding = np.zeros((len(unique_entity_list), 50), dtype = np.float32)
for i, entity in enumerate(unique_entity_list):
    cluster_id = sub_graph_entity_idx_to_cluster_id_dict[int(i)]
    if cluster_id != -1:
        idx = cluster_kg_embedding_indices.index(cluster_id)
        emd = cluster_kg_embedding[idx]
        rearranged_cluster_kg_entity_embedding[i] = emd

# concatenate sub-graph entity embedding and cluster KG embedding
concatenated_embedding = np.concatenate((rearranged_sub_graph_entity_embedding, rearranged_cluster_kg_entity_embedding), axis = 1)

# save concatenated embedding
np.save(concatenated_embedding, 'concatenated_embedding_'+ds_name+'.npy')



    




