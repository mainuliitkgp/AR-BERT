import urllib.parse, urllib.request, json
import sys
import xlwt 
from xlwt import Workbook
import pickle

def CallWikifier(text, ch_f, ch_t, lang="en", threshold=0.8):
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", "nxykvtsjsoztxdkkxhzqmejvifuqit"),
        ("pageRankSqThreshold", "%g" % threshold), ("applyPageRankSqThreshold", "true"),
        ("wikiDataClasses", "false"), ("wikiDataClassIds", "false"),
        ("support", "true"), ("ranges", "false"),
        ("includeCosines", "true")
        ])
    url = "http://www.wikifier.org/annotate-article"
    # Call the Wikifier and read the response.
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout = 60) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))    

    if len(response["annotations"]) == 0:
        return []
    else return response["annotations"]
    

wikifier_input_data_dir = sys.argv[1]
dataset = sys.argv[2]
mode = sys.argv[3]
wikifier_output_data_dir = sys.argv[4]

fp = open(wikifier_input_data_dir+'/'+dataset+'/'+mode+'_sent.txt', 'r')
sentences = fp.readlines()
fp.close()

fp_asp = open(wikifier_input_data_dir+'/'+dataset+'/'+mode+'_asp.txt', 'r')
aspects = fp_asp.readlines()
fp_asp.close()

fp_idx = open(wikifier_input_data_dir+'/'+dataset+'/'+mode+'_idx.txt', 'r')
idx = fp_idx.readlines()
fp_idx.close()

output_list = []

for i in range(len(sentences)):
    sent = sentences[i].strip()
    aspect = aspects[i].strip()
    indices = idx[i].strip().split()
    ch_f = int(indices[0])
    ch_t = int(indices[1])-1
    output_list.append(CallWikifier(sent, ch_f, ch_t, lang="en", threshold=1.0))

with open(wikifier_output_data_dir+'/'+dataset+'/'+mode+'_wikification_annotation_list.pkl', 'wb') as fp_pickle:
    pickle.dump(output_list, fp_pickle)

    
