import scanpy as sc
import torch
import sys
sys.path.append('/guoxiaopeng/wangjue/VQCell/FoldToken2')
sys.path.append('/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4')
import pandas as pd
from torch_scatter import scatter_mean
import json
import numpy as np

data_path = 'feature.csv'
genes = pd.read_csv(data_path)['feature_name'].tolist()
genes = [gene for i, gene in enumerate(genes) if 'RP' not in gene and gene != 'MALAT1']

attention, vq_code, gene_indexes = torch.load('blood.pth')
print(np.unique(vq_code))
attention = scatter_mean(attention.cuda(), index=torch.from_numpy(gene_indexes).cuda(), dim=1)
code_attention = scatter_mean(attention.cuda(), index=torch.from_numpy(vq_code).cuda(), dim=0)

values, indexes = torch.topk(code_attention, k=10)
w = open('blood.jsonl', 'w')
indexes = indexes.tolist()

for i in range(code_attention.shape[0]):
    res = {}
    res['vq_code'] = i 
    res['genes'] = [genes[index] for index in indexes[i]]
    res['values'] = values[i].tolist()
    w.write(json.dumps(res) + '\n')




