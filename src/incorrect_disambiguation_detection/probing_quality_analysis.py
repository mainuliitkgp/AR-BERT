import sys
import pickle
from matplotlib import pyplot as plt
import xlwt
from xlwt import Workbook

semantic_res_fp = sys.argv[1]
structural_res_fp = sys.argv[2]
unique_entity_fp = sys.argv[3]
ds_name = sys.argv[4]
raw_sent_fp = sys.argv[5]
raw_asp_fp = sys.argv[6]


# Workbook is created 
wb = Workbook()
# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1')
# Specifying style 
style = xlwt.easyxf('font: bold 1')
col_names = ['sentence', 'aspect', 'entity', 'semantic probe score', 'structural probe score']
for i, col_name in enumerate(col_names):
    sheet1.write(0, i, col_name, style)
wb.save(ds_name+'_probing_quality_analysis.xls')

sheet_row_idx = 1


# read probe output
# semantic
with open(semantic_res_fp, 'rb') as fp:
    semantic_res_list = pickle.load(fp)

# structural
with open(structural_res_fp, 'rb') as fp:
    structural_res_list = pickle.load(fp)


# prune probe output above >= 0
semantic_res_list_pruned, structural_res_list_pruned = {}, {}
# semantic
for i in range(len(semantic_res_list)):
    if semantic_res_list[i][1] >= 0.0 :
        semantic_res_list_pruned[int(semantic_res_list[i][0])] = 1
    else:
        semantic_res_list_pruned[int(semantic_res_list[i][0])] = 0

# structural
for i in range(len(structural_res_list)):
    if structural_res_list[i][1] >= 0.0 :
        structural_res_list_pruned[int(structural_res_list[i][0])] = 1
    else:
        structural_res_list_pruned[int(structural_res_list[i][0])] = 0

# prepare unique entity list
unique_entity_list = []
with open(unique_entity_fp) as fp:
    for line in fp:
        unique_entity_list.append(line.strip())

# prepare unique entity list
unique_entity_list = []
with open(unique_entity_fp) as fp:
    for line in fp:
        unique_entity_list.append(line.strip())

# read sent idx to entity idx in dataset
with open(sent_idx_to_entity_idx_fp, 'rb') as fp:
    sent_idx_to_entity_idx_list = pickle.load(fp)

sent, asp = [], []
with open(raw_sent_fp) as fp:
    for line in fp:
        sent.append(line.strip())

with open(raw_asp_fp) as fp:
    for line in fp:
        asp.append(line.strip())

for i in range(len(sent)):
    row_list = []
    row_list = [sent[i], asp[i], unique_entity_list[sent_idx_to_entity_idx_list[i]], semantic_res_list_pruned[i], structural_res_list_pruned]
    sheet1.write(sheet_row_idx, col_idx)









