import sys
from nltk import ngrams
from nltk.tag import pos_tag
import tokenization

raw_sent_fp = sys.argv[1]
raw_asp_fp = sys.argv[2]
vocab_file = sys.argv[3]
ds_name = sys.argv[4]

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

count = 0
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

    ch_len = 0
    for item in sent[0:start]:
        ch_len += len(item)
    if ch_len + len(sent[0:start]) != raw_sent_list[i].strip().index(raw_asp_list[i].strip()):
        count += 1
    

print(count)
        
