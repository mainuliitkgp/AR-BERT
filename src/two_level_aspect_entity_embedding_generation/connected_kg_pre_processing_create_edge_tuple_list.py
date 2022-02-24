import json
import pickle
import sys

edge_tuple_list = [] # list containg edges in tuple format
entity_to_idx ={} # entity name to id dict
idx_to_entity ={} # id to entity name dict
count = 0 # counter indicating number of processed edges
id_count = 0 # entity id starts with 0

with open('dbpedia_pagelink_entity_pair_list_max_connected_component.txt') as fp:
    for line in fp:
        words = line.strip().split()
        key1 = words[0]
        key2 = words[1]
        if key1 not in entity_to_idx:
            entity_to_idx[key1] = id_count
            idx_to_entity[id_count] = key1
            id_count = id_count+1 
        if key2 not in entity_to_idx:
            entity_to_idx[key2] = id_count
            idx_to_entity[id_count] = key2
            id_count = id_count+1
        edge_tuple_list.append((entity_to_idx[key1], entity_to_idx[key2])) # edge in tuple format
        count = count+1
        if count% 100000 == 0:
            print('Done with '+str(count))

print('Done with edge tuple list')

with open('connected_dbpedia_pagelink_edge_tuple_list.pkl', 'wb') as fp1:
    pickle.dump(edge_tuple_list, fp1)

fp4 = open('connected_graph.txt', 'w')

for item in edge_tuple_list:
    fp4.write(str(item[0])+' '+str(item[1])+'\n')

fp4.close()

del(edge_tuple_list)

with open('connected_entity_to_idx_dbpedia_pagelink.json', 'w') as fp2:
    json.dump(entity_to_idx, fp2)

del(entity_to_idx)

with open('connected_idx_to_entity_dbpedia_pagelink.json', 'w') as fp3:
    json.dump(idx_to_entity, fp3)

del(idx_to_entity)

print('Done')


    



