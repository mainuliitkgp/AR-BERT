import sys
from nltk import ngrams
from nltk.tag import pos_tag
import tokenization
import json
import numpy as np
import random
import pickle 
 
raw_sent_fp = sys.argv[1]
raw_asp_fp = sys.argv[2]
raw_idx_fp = sys.argv[3]
vocab_file = sys.argv[4]
bert_tokens_embd_fp = sys.argv[5]
sampled_op_fp = sys.argv[6]
entity_idx_to_list_of_sent_idx_fp = sys.argv[7]
out_dir = sys.argv[8]
ds_name = sys.argv[9]

# read data
# sent
raw_sent_list = []
with open(raw_sent_fp) as fp:
    for line in fp:
        raw_sent_list.append(line.strip())

# aspect
raw_asp_list = []
with open(raw_asp_fp) as fp:
    for line in fp:
        raw_asp_list.append(line.strip())

# indices
raw_idx_list = []
with open(raw_idx_fp) as fp:
    for line in fp:
        words = line.strip().split()
        ch_f = int(words[0])
        ch_t = int(words[1])
        raw_idx_list.append((ch_f, ch_t))

# read bert tokens embedding
fo = open(bert_tokens_embd_fp)
bert_tokens_embd_lines = fo.readlines()

# tokenize and map aspect in sentence to tokenized sentence
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

count = 0
aspect_token_indices_list = []
aspect_bert_embd_all_sent = []
for i in range(len(raw_sent_list)):
    sent = raw_sent_list[i].strip()
    asp = raw_asp_list[i].strip()
    asp_ch_f = raw_idx_list[i][0]
    asp_ch_t = raw_idx_list[i][1]

    left_sent = sent[:asp_ch_f].strip()
    right_sent = sent[asp_ch_t:].strip()

    sent_tokenized_list = tokenizer.tokenize(sent)
    left_sent_tokenized_list = tokenizer.tokenize(left_sent)
    asp_tokenized_list = tokenizer.tokenize(asp)
    right_sent_tokenized_list = tokenizer.tokenize(right_sent)

    #if sent_tokenized_list[:len(left_sent_tokenized_list)] == left_sent_tokenized_list and sent_tokenized_list[len(left_sent_tokenized_list):len(left_sent_tokenized_list)+len(asp_tokenized_list)] == asp_tokenized_list and sent_tokenized_list[len(left_sent_tokenized_list)+len(asp_tokenized_list):] == right_sent_tokenized_list:
    #    count += 1
        
    
    aspect_token_indices_list.append((len(left_sent_tokenized_list), (len(left_sent_tokenized_list)+len(asp_tokenized_list)-1)))

    embd_for_all_tokens_in_sent_dict = json.loads(bert_tokens_embd_lines[i])
    embd_for_all_tokens_in_sent_wo_CLS_and_SEP = embd_for_all_tokens_in_sent_dict['features'][1:-1]
    avg_aspect_embd = np.zeros((768,), dtype = np.float32)
    bert_tokens_list = []
    for token_rep in embd_for_all_tokens_in_sent_wo_CLS_and_SEP[len(left_sent_tokenized_list):len(left_sent_tokenized_list)+len(asp_tokenized_list)]:
        bert_tokens_list.append(token_rep['token'])
        all_layer_rep_list = token_rep['layers']
        avg_layer_emd = np.zeros((768,), dtype = np.float32)
        for layer_rep in all_layer_rep_list:
            avg_layer_emd += np.array(layer_rep['values'], dtype = np.float32)
        avg_layer_emd = avg_layer_emd / len(all_layer_rep_list)

        avg_aspect_embd += avg_layer_emd

    avg_aspect_embd = avg_aspect_embd / len(embd_for_all_tokens_in_sent_wo_CLS_and_SEP[len(left_sent_tokenized_list):len(left_sent_tokenized_list)+len(asp_tokenized_list)])


    aspect_bert_embd_all_sent.append(avg_aspect_embd)

    if bert_tokens_list != asp_tokenized_list:
        count += 1

aspect_bert_embd_all_sent = np.array(aspect_bert_embd_all_sent, dtype = np.float32)

print(count) 
    
del(bert_tokens_embd_lines)

# read sampled output nodes i, j, k in graph space-entity idx
with open(sampled_op_fp, 'rb') as fp:
    sampled_output_list = pickle.load(fp)

# read entity index to list of sentence index
entity_idx_to_list_of_sent_idx_dict = json.load(open(entity_idx_to_list_of_sent_idx_fp))

# prepare bert embedding for nodes i, j and k
result_embd = []
for i in range(len(raw_sent_list)):
    node_i_embd = aspect_bert_embd_all_sent[i]

    node_j = sampled_output_list[i][1]
    sent_idx_list_for_node_j = entity_idx_to_list_of_sent_idx_dict[str(node_j)]
    sampled_sent_idx_j = random.choice(sent_idx_list_for_node_j)
    node_j_embd = aspect_bert_embd_all_sent[sampled_sent_idx_j]

    node_k = sampled_output_list[i][2]
    sent_idx_list_for_node_k = entity_idx_to_list_of_sent_idx_dict[str(node_k)]
    sampled_sent_idx_k = random.choice(sent_idx_list_for_node_k)
    node_k_embd = aspect_bert_embd_all_sent[sampled_sent_idx_k]

    concatenated_i_j_embd = np.concatenate((node_i_embd, node_j_embd), axis = -1)
    concatenated_i_j_k_embd = np.concatenate((concatenated_i_j_embd, node_k_embd), axis = -1)

    result_embd.append(concatenated_i_j_k_embd)

result_embd = np.array(result_embd, dtype = np.float32)
np.save(out_dir+ds_name+'_bert_embedding_nodes_i_j_k.npy', result_embd)

