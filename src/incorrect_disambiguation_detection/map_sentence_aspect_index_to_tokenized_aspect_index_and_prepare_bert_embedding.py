import sys
from nltk import ngrams
from nltk.tag import pos_tag
import tokenization

raw_sent_fp = sys.argv[1]
raw_asp_fp = sys.argv[2]
ds_name = sys.argv[3]
vocab_file = sys.argv[4]

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

# tokenize and map aspect in sentence to tokenized sentence
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True) 

asp_indices_in_sent = []
for i in range(len(raw_sent_list)):
    sent = raw_sent_list[i].strip().split()
    asp = raw_asp_list[i].strip().split()

    for s_i in range(len(sent)):
        flag = True
        for a_j in range(len(asp)):
            if ds_name == 'twitter':
                tag = pos_tag(asp[a_j].split())
                if tag[0][1]=='NN' or tag[0][1]=='NNS' or tag[0][1]=='NNP' or tag[0][1]=='NNPS':
                    asp[a_j] = asp[a_j].title() 
            if sent[s_i+a_j] == asp[a_j] or asp[a_j] in sent[s_i+a_j]:
                start = s_i
                continue
            else:
                flag = False
                break 

        if flag: 
            break

    if len(asp) == 1:
        asp_indices_in_sent.append([start])
    else:
        asp_indices_in_sent.append([start, (start+len(asp)-1)])

    ### Output
    bert_tokens = []

    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.
    orig_to_tok_map = []

    bert_tokens.append("[CLS]")
    for orig_token in sent:
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    bert_tokens.append("[SEP]")

    map_from, map_to = 0, 0 
    if len(asp) == 1:
        map_from = orig_to_tok_map[start]-1
        map_to = orig_to_tok_map[start+1]-2
    else:
        map_from = orig_to_tok_map[start]-1
        map_to = orig_to_tok_map[(start+len(asp)-1)+1]-2

    print(tokenizer.tokenize(raw_sent_list))
    print(asp)
    print(map_from, map_to)
    print('\n')
        


            

 


