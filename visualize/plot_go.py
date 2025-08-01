import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

np.random.seed(42)

grn = torch.load('datas/grn.pth')
go = torch.load('datas/go.pth')

gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = gene_list_df['gene_name']
gene_l = gene_list.to_list()

file = open('gene.txt').readlines()
genes = []
for line in file:
    if not line.startswith('Mm'):
        continue
    gene = line.strip().split('\t')[1]
    if gene in gene_l:
        genes.append(gene)


indexes = [gene_l.index(gene) for gene in genes]
print(indexes)

indexes1 = indexes[:20]
indexes2 = indexes

print(1)
go = go[indexes1][:, indexes2]
go[go>0.3] = 1
go[go<=0.3] = 0

plt.figure(figsize=(32, 12))
sns.heatmap(go, annot=False, cmap='Blues', xticklabels=gene_list[indexes2].tolist(), yticklabels=gene_list[indexes1].tolist())
plt.title('Gene-gene interaction predicted by ChrisCell', fontsize=22)
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(rotation=0, fontsize=15)

plt.tight_layout()

plt.savefig('figures/go.pdf')