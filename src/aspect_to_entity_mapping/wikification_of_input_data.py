import urllib.parse, urllib.request, json
import sys
import xlwt 
from xlwt import Workbook

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

    annotation_list = []
    output_list = []

    dbpedia_link = ''
    wikipedia_link = ''
    max_len = 0

    if len(response["annotations"]) == 0:
        return []
    
    for annotation in response["annotations"]:
        for item in annotation["support"]:
            if(item['chFrom']==ch_f and item['chTo']==ch_t):

                annotation_list = []
                output_list = []

                annotation_list.extend([annotation['title'], annotation['url'], annotation['pageRank'], annotation['cosine'], annotation['dbPediaIri'], item['chFrom'], item['chTo'], item['pMentionGivenSurface'], item['pageRank'], item['prbConfidence'], item['entropy'], annotation['url'], annotation['dbPediaIri']])
                output_list.append(annotation_list)

                return output_list

            if(item['chFrom']==ch_f and item['chTo']<ch_t) or (item['chFrom']>ch_f and item['chTo']==ch_t) or (item['chFrom']>ch_f and item['chTo']<ch_t):
                annotation_list = [] 
                annotation_list.extend([annotation['title'], annotation['url'], annotation['pageRank'], annotation['cosine'], annotation['dbPediaIri'], item['chFrom'], item['chTo'], item['pMentionGivenSurface'], item['pageRank'], item['prbConfidence'], item['entropy']])
                output_list.append(annotation_list) 
                
            if(item['chFrom']==ch_f and item['chTo']<ch_t):
                if item['chTo']-item['chFrom']+1 > max_len:
                    max_len = item['chTo']-item['chFrom']+1
                    wikipedia_link = annotation['url']
                    dbpedia_link = annotation['dbPediaIri']

            if(item['chFrom']>ch_f and item['chTo']==ch_t):
                if item['chTo']-item['chFrom']+1 > max_len:
                    max_len = item['chTo']-item['chFrom']+1
                    wikipedia_link = annotation['url']
                    dbpedia_link = annotation['dbPediaIri']

            if(item['chFrom']>ch_f and item['chTo']<ch_t):
                if item['chTo']-item['chFrom']+1 > max_len:
                    max_len = item['chTo']-item['chFrom']+1
                    wikipedia_link = annotation['url']
                    dbpedia_link = annotation['dbPediaIri']
    
    if max_len > 0:
        for i in range(len(output_list)):
            output_list[i].extend([wikipedia_link, dbpedia_link])
        
        return output_list
    else:
        return []

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

fp_entity = open(wikifier_output_data_dir+'/'+dataset+'/'+mode+'_wikified_entity_'+dataset+'.txt', 'w')

# Workbook is created 
wb = Workbook()   
# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1')
# Specifying style 
style = xlwt.easyxf('font: bold 1')
col_names = ['sentence', 'aspect', 'aspect_ch_from', 'aspect_ch_to', 'title', 'url', 'pagerank', 'cosine', 'dbpediaIri', 'support_chFrom', 'support_chTo', 'pMentionGivenSurface', 'support_pagerank', 'prbConfidence', 'entropy', 'wikipedia_link', 'dbpedia_link']
for i, col_name in enumerate(col_names):
    sheet1.write(0, i, col_name, style)
wb.save(wikifier_output_data_dir+'/'+dataset+'/'+mode+'_wikified_annotation_'+dataset+'.xls') 

sheet_row_idx = 1

for i in range(len(sentences)):
    sent = sentences[i].strip()
    aspect = aspects[i].strip()
    indices = idx[i].strip().split()
    ch_f = int(indices[0])
    ch_t = int(indices[1])-1
    output_list = CallWikifier(sent, ch_f, ch_t, lang="en", threshold=1.0)

    if len(output_list) == 0:
        output_list.extend([sent.strip(), aspect.strip(), ch_f, ch_t, 'Unknown', 'Unknown', '-', '-', 'Unknown',  '-', '-', '-', '-', '-', '-', 'Unknown', 'Unknown'])
        for col_idx in range(len(output_list)):
            sheet1.write(sheet_row_idx, col_idx, output_list[col_idx])
        sheet_row_idx += 1

        for col_idx in range(len(output_list)):
            sheet1.write(sheet_row_idx, col_idx, ' ')
        sheet_row_idx += 1

        fp_entity.write('Unknown Unknown\n')

    else:
        for idx_count in range(len(output_list)):
            output_list[idx_count] = [sent.strip(), aspect.strip(), ch_f, ch_t] + output_list[idx_count]

        for j in range(len(output_list)):
            for k in range(len(output_list[j])):
                sheet1.write(sheet_row_idx, k, output_list[j][k])
            sheet_row_idx += 1

        for col_index in range(len(col_names)):
            sheet1.write(sheet_row_idx, col_index, ' ')
        sheet_row_idx += 1

        fp_entity.write(output_list[0][-2]+' '+output_list[0][-1]+'\n')

    if i % 100 == 0:
        wb.save(wikifier_output_data_dir+'/'+dataset+'/'+mode+'_wikified_annotation_'+dataset+'.xls')

wb.save(wikifier_output_data_dir+'/'+dataset+'/'+mode+'_wikified_annotation_'+dataset+'.xls')

fp_entity.close()
    
