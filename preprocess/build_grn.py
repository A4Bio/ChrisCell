import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

grn = pd.read_csv('grn.csv')
gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])

attn = torch.zeros((len(gene_list), len(gene_list)))

for i in tqdm(range(len(grn))):
    gene1 = grn['source'][i]
    gene2 = grn['target'][i]
    value = grn['importance'][i]
    if gene1 in gene_list and gene2 in gene_list:
        attn[gene_list.index(gene1)][gene_list.index(gene2)] = value
        attn[gene_list.index(gene2)][gene_list.index(gene1)] = value

torch.save(attn, 'grn.pth')
