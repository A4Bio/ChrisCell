import scanpy as sc
import torch
import sys
sys.path.append('/guoxiaopeng/wangjue/VQCell/FoldToken2')
sys.path.append('/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4')
import pandas as pd
from torch_scatter import scatter_mean
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_path = 'feature.csv'
genes = pd.read_csv(data_path)['feature_name'].tolist()

attention, vq_code, gene_indexes = torch.load('brain.pth')
attention = scatter_mean(attention, index=torch.from_numpy(gene_indexes).cuda(), dim=1)
vq_code_uni, code_indexes = torch.unique(torch.from_numpy(vq_code), return_inverse=True)
code_attention = scatter_mean(attention, index=code_indexes.cuda(), dim=0)

values, indexes = torch.topk(code_attention, k=10)
w = open('brain.jsonl', 'w')
indexes = indexes.tolist()

all_res = []
all_genes = []
for i in range(len(vq_code_uni)):
    res = {}
    res['vq_code'] = int(vq_code_uni[i]) 
    res['genes'] = [genes[index] for index in indexes[i]][:5]
    res['values'] = values[i].tolist()[:5]
    all_genes+=res['genes']
    all_res.append(res)
all_genes = sorted(list(set(all_genes)))
all_genes.pop(all_genes.index('MALAT1'))
cm = np.zeros([len(vq_code_uni), len(all_genes)])

for i, res in enumerate(all_res):
    for j, gene in enumerate(all_genes):
        if gene in res['genes']:
            index = res['genes'].index(gene)
            cm[i][j] = res['values'][index]

plt.figure(figsize=(32, 12))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=all_genes, yticklabels=vq_code_uni.tolist())
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()

plt.savefig('figures/code_gene.pdf')








