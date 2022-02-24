import xml.etree.ElementTree as ET
import sys
from nltk import ngrams
from nltk.tag import pos_tag

# raw data file path
data_path = sys.argv[1]

# wikifier data input directory
wiki_data_dir = sys.argv[2]

dataset = sys.argv[3]

mode = sys.argv[4] #train or test

polar_idx={'positive': 0, 'negative': 1, 'neutral': 2}
idx_polar={0: 'positive', 1: 'negative', 2: 'neutral'}

def parse_SemEval14(fn):
    sent_list = []
    asp_list = []
    fr_to_list = []
    root=ET.parse(fn).getroot()
    corpus=[]
    opin_cnt=[0]*len(polar_idx)
    for sent in root.iter("sentence"):
        opins=set()
        for opin in sent.iter('aspectTerm'):
            if int(opin.attrib['from'] )!=int(opin.attrib['to'] ) and opin.attrib['term']!="NULL":
                if opin.attrib['polarity'] in polar_idx:
                    sent_list.append(sent.find('text').text)
                    asp_list.append(opin.attrib['term'])
                    fr_to_list.append((int(opin.attrib['from']), int(opin.attrib['to'])))

    return sent_list, asp_list, fr_to_list

def read_raw_twitter(fn):
        sent_list = []
        asp_list = []
        fr_to_list = []
        with open(fn) as fp:
            for line in fp:
                tokens = line.strip().split()
                words, target_words = [], []
                indices=[]
                index=-1
                tar_str=''
                wiki_inp_str=''
                for t in tokens:
                    index=index+1
                    if(t.endswith('/p') or t.endswith('/n') or t.endswith('/0')) and (t!='y/n'):
                        # negative: 0, positive: 1, neutral: 2
                        # note, this part should be consistent with evals part
                        end = 'xx'
                        y = 0
                        if '/p' in t:
                            end = '/p'
                            y = 1
                        elif '/n' in t:
                            end = '/n'
                            y = 0
                        elif '/0' in t:
                            end = '/0'
                            y = 2
                        words.append(t[:-len(end)])
                        target_words.append(t[:-len(end)])
                        indices.append(index)
                        tag = pos_tag(t[:-len(end)].split())
                        if tag[0][1]=='NN' or tag[0][1]=='NNS' or tag[0][1]=='NNP' or tag[0][1]=='NNPS': 
                            wiki_inp_str=wiki_inp_str+' '+t[:-len(end)].title()
                        else:
                            wiki_inp_str=wiki_inp_str+' '+t[:-len(end)]

                    else:
                        words.append(t)
                        wiki_inp_str=wiki_inp_str+' '+t


                for target in target_words:
                    tar_str=tar_str+target+end+' '
                tar_str=tar_str.strip()

                ch_from=line.find(tar_str)
                ch_to = (ch_from+len(tar_str)-1-len(target_words)*2) + 1

                sent_list.append(wiki_inp_str.strip())

                tar_str_wo_end = ''
                for target in target_words:
                    tar_str_wo_end = tar_str_wo_end+target+' '
                asp_list.append(tar_str_wo_end.strip())

                fr_to_list.append((int(ch_from), int(ch_to)))
                
        return sent_list, asp_list, fr_to_list 


if dataset == 'twitter':
    sent_list, asp_list, fr_to_list = read_raw_twitter(data_path)
else:
    sent_list, asp_list, fr_to_list = parse_SemEval14(data_path)

fp_sent = open(wiki_data_dir+dataset+'/'+mode+'_sent.txt', 'w')
fp_asp = open(wiki_data_dir+dataset+'/'+mode+'_asp.txt', 'w')
fp_idx = open(wiki_data_dir+dataset+'/'+mode+'_idx.txt', 'w')

for i in range(len(sent_list)):
    fp_sent.write(sent_list[i].strip()+'\n')
    fp_asp.write(asp_list[i].strip()+'\n')
    fp_idx.write(str(fr_to_list[i][0])+' '+str(fr_to_list[i][1])+'\n')

fp_sent.close()
fp_asp.close()
fp_idx.close() 




