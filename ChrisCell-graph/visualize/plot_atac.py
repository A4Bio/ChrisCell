import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import torch
from torch_scatter import scatter_mean

data_path = '/guoxiaopeng/wangjue/data/celldata/simba_data/atac_seq_fig4.h5ad'
data0 = sc.read(data_path)

attention, vq_code, gene_indexes = torch.load('atac.pth')
index342 = vq_code==342
index16 = vq_code==16
index560 = vq_code==560
index913 = vq_code==913

attention_ = scatter_mean(attention[index342].cuda(), index=gene_indexes[index342].cuda(), dim=1).mean(0)
value, index = torch.topk(attention_, k=5)
index = index.tolist()
chrs = [data0.var['Gene'].keys().tolist()[i] for i in index]
genes = [data0.var['Gene'][i] for i in index]
print(342, chrs, genes)

attention_ = scatter_mean(attention[index16].cuda(), index=gene_indexes[index16].cuda(), dim=1).mean(0)
value, index = torch.topk(attention_, k=5)
index = index.tolist()
chrs = [data0.var['Gene'].keys()[i] for i in index]
genes = [data0.var['Gene'][i] for i in index]
print(16, chrs, genes)

attention_ = scatter_mean(attention[index560].cuda(), index=gene_indexes[index560].cuda(), dim=1).mean(0)
value, index = torch.topk(attention_, k=5)
index = index.tolist()
chrs = [data0.var['Gene'].keys()[i] for i in index]
genes = [data0.var['Gene'][i] for i in index]
print(560, chrs, genes)

attention_ = scatter_mean(attention[index913].cuda(), index=gene_indexes[index913].cuda(), dim=1).mean(0)
value, index = torch.topk(attention_, k=5)
index = index.tolist()
chrs = [data0.var['Gene'].keys()[i] for i in index]
genes = [data0.var['Gene'][i] for i in index]
print(913, chrs, genes)