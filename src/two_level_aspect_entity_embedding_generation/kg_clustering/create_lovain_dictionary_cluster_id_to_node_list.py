import pickle
import sys

# create {node:cluster_id} dictionary and cluster dictionary {cluster_id:[node]}
node_to_cluster_id_dict = {}
cluster_dict = {}

node_cluter_pair_file_path = sys.argv[1]

with open(node_cluter_pair_file_path) as fp:
    for line in fp:
        words = line.strip().split()
        if int(words[1]) not in cluster_dict:
            cluster_dict[int(words[1])] = []
        cluster_dict[int(words[1])].append(int(words[0]))
       
        node_to_cluster_id_dict[int(words[0])] = int(words[1])

with open('cluster_dict.pkl', 'wb') as fp:
    pickle.dump(cluster_dict, fp)

with open('node_to_cluster_id_dict.pkl', 'wb') as fp:
    pickle.dump(node_to_cluster_id_dict, fp)  
        


    
