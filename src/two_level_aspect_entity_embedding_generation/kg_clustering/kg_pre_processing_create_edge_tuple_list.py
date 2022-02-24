import json
import pickle
import sys

edge_tuple_list = [] # list containg edges in tuple format
entity_to_idx ={} # entity name to id dict
idx_to_entity ={} # id to entity name dict
count = 0 # counter indicating number of processed edges
id_count = 0 # entity id starts with 0

kg_dir = sys.argv[1]

with open(kg_dir+'page_links_en.ttl') as fp:
    for line in fp:
        if line.strip().startswith('<'):
            words = line.strip().split('>')
            key1 = words[0].strip().replace('<http://dbpedia.org/resource/', '')
            key2 = words[2].strip().replace('<http://dbpedia.org/resource/', '')
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

with open('dbpedia_pagelink_edge_tuple_list.pkl', 'wb') as fp1:
    pickle.dump(edge_tuple_list, fp1)

del(edge_tuple_list)

with open('entity_to_idx_dbpedia_pagelink.json', 'w') as fp2:
    json.dump(entity_to_idx, fp2)

del(entity_to_idx)

with open('idx_to_entity_dbpedia_pagelink.json', 'w') as fp3:
    json.dump(idx_to_entity, fp3)

del(idx_to_entity)

print('Done')


    



