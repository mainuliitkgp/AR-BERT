import sys
import numpy as np
import math
import pickle
import random
import json

unique_entity_fp = sys.argv[1]
train_raw_fp = sys.argv[2]
test_raw_fp = sys.argv[3]
ds_name = sys.argv[4]
k = sys.argv[5] # sampling limit

# prepare unique entity list
unique_entity_list = []
with open(unique_entity_fp) as fp:
    for line in fp:
        unique_entity_list.append(line.strip())

# read raw data and map sentence aspect index to entity index
# train
train_sent_asp_to_entity_idx = []
with open(train_raw_fp) as fp:
    for line in fp:
        train_sent_asp_to_entity_idx.append(unique_entity_list.index(line.strip()))

# test
test_sent_asp_to_entity_idx = []
with open(test_raw_fp) as fp:
    for line in fp:
        test_sent_asp_to_entity_idx.append(unique_entity_list.index(line.strip()))


sent_asp_to_entity_idx = train_sent_asp_to_entity_idx + test_sent_asp_to_entity_idx

# map entity idx to list of sentence-aspect idx
entity_idx_to_list_of_sent_asp_idx = {}

for i, item in enumerate(sent_asp_to_entity_idx):
    if int(item) not in entity_idx_to_list_of_sent_asp_idx:
        entity_idx_to_list_of_sent_asp_idx[int(item)] = []
    entity_idx_to_list_of_sent_asp_idx[int(item)].append(i)

# sampling
with open(ds_name+'_soreted_node_list_wrt_sigmoid_dot.pkl', "rb") as fp:
    soreted_node_list_wrt_sigmoid_dot = pickle.load(fp)

sampling_res = []
for sent_idx in range(len(sent_asp_to_entity_idx)):
    entity_idx = sent_asp_to_entity_idx[sent_idx]
    entity_list = soreted_node_list_wrt_sigmoid_dot[entity_idx]
    top_k_entity_list = entity_list[0:int(k)]
    bottom_k_entity_list = entity_list[-int(k):]
    
    nearest_entity = random.choice(top_k_entity_list)
    farthest_entity = random.choice(bottom_k_entity_list)
    sampling_res.append((entity_idx, nearest_entity, farthest_entity))

print(sampling_res)

with open(ds_name+'_sent_asp_to_entity_idx.pkl', 'wb') as fp:
    pickle.dump(sent_asp_to_entity_idx, fp)

with open(ds_name+'_entity_idx_to_list_of_sent_asp_idx.json', 'w') as fp:
    json.dump(entity_idx_to_list_of_sent_asp_idx, fp)

with open(ds_name+'_sampling_res.pkl', 'wb') as fp:
    pickle.dump(sampling_res, fp)


 
