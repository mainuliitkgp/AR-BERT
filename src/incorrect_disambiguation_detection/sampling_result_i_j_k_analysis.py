import sys
import pickle

unique_entity_fp=sys.argv[1]
sampled_list_fp = sys.argv[2]
output_dir = sys.argv[3]
ds_name = sys.argv[4]

unique_entity_list=[]
with open(unique_entity_fp) as fp:
    for line in fp:
        unique_entity_list.append(line.strip())

with open(sampled_list_fp, "rb") as fp:
    sampled_list=pickle.load(fp)

fp = open(output_dir+ds_name+'_sampling_result_nodes_i_j_k.txt')
for item in sampled_list:
    fp.write('node: 'unique_entity_list[item[0]]+'    closer node: '+unique_entity_list[item[1]]+'    far away node:'+unique_entity_list[item[2]])
