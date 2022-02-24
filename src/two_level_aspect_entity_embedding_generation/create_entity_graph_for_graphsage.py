import json
import numpy as np
import os
import pickle
import sys

dataset_name = sys.argv[1]

def build_vocab(target_list):
        vocab = {}
        idx = 1  # start from 1
        n_records = len(target_list)
        for i in range(n_records):
            wl = []
            words = target_list[i].strip().split('/')
            for word in words:
                wl.extend(word.strip().split('_'))
            for w in wl:
                if w.startswith('('):
                    w=w[len('('):]
                if w.endswith(')'):
                    w=w[:-len(')')]
                if w not in vocab:
                    vocab[w] = idx
                    idx += 1
        return vocab

def get_embedding(vocab):
        emb_file = '/home/mainul/ACL_2020/TNet_Author_code_Original/embeddings/glove_840B_300d.txt'   # path of the pre-trained word embeddings
        #pkl = '/home/mainul/EMNLP_2020/kg_embedding/connected_kg_entities_840B.pkl'  # word embedding file of the current dataset

        print("Load embeddings from %s ..." % (emb_file))
        n_emb = 0
        #if not os.path.exists(pkl):
        embeddings = np.zeros((len(vocab)+1, 300), dtype='float32')
        with open(emb_file) as fp:
            for line in fp:
                eles = line.strip().split()
                w = eles[0]
                n_emb += 1
                if w in vocab:
                    try:
                        embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                    except ValueError:
                        pass
            print("Find %s word embeddings!!" % n_emb)
            #pickle.dump(embeddings, open(pkl, 'wb'))
        #else:
            #embeddings = pickle.load(open(pkl, 'rb'))
        for i in range(len(embeddings)):
            if i and np.count_nonzero(embeddings[i]) == 0:
                embeddings[i] = np.random.uniform(-0.25, 0.25, embeddings.shape[1])
        return embeddings

## mapped target to node id
target_to_id={}
node_id=0
with open(dataset_name+'_unique_entities.txt') as fp:
    for line in fp:
        target_to_id[line.strip()]=node_id
        node_id=node_id+1

## creating nodes
nodes=[]
node={}
for i in range(len(target_to_id)):
    node['test']=False
    node['id']=i
    node['val']=False
    nodes.append(node)
    node={}

## creating edges
edges=[]
edge={}
with open(dataset_name+'_entity_edges.txt') as fp:
    for line in fp:
        entities = line.strip().split()
        node1 = target_to_id[entities[0]]
        node2 = target_to_id[entities[1]]
        edge['source']=min(node1, node2)
        edge['target']=max(node1, node2)
        edges.append(edge)
        edge={}

## creating graph
G={}
G['directed']=False
G['graph']={}
G['nodes']=nodes
G['links']=edges
G['multigraph']=False

with open(dataset_name+'-G.json', 'w') as fp:
    json.dump(G, fp)

## creating node_id to id integer
node_id_dict={}
for i in range(len(target_to_id)):
    node_id_dict[str(i)]=i

with open(dataset_name+'-id_map.json', 'w') as fp:
    json.dump(node_id_dict, fp)

## creating node_id to classes
node_class={}
class_list=np.random.randint(0,3,len(target_to_id))
for i in range(len(class_list)):
    class_label=np.zeros(3,np.int8)
    class_label[class_list[i]]=1
    class_label=class_label.tolist()
    node_class[str(i)]=class_label

with open(dataset_name+'-class_map.json', 'w') as fp:
    json.dump(node_class, fp)
	
## creating node_features
target_list=list(target_to_id.keys())
vocab=build_vocab(target_list)
embedding_vocab = get_embedding(vocab)

arr_of_emb=[]

with open(dataset_name+'_unique_entities.txt') as fp:
    for line in fp:
        wl = []
        words = line.strip().split('/')
        for word in words:
            wl.extend(word.strip().split('_'))
        emb=np.zeros(300)
        for w in wl:
            if w.startswith('('):
                w=w[len('('):]
            if w.endswith(')'):
                w=w[:-len(')')]
            emb = emb + embedding_vocab[vocab[w]]
        emb=emb/len(wl)
        arr_of_emb.append(emb)

arr_of_emb = np.array(arr_of_emb, dtype='float32')
np.save(dataset_name+'-feats.npy',arr_of_emb)




