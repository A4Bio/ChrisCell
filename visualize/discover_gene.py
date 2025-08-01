import scanpy as sc
import torch
import sys
sys.path.append('/guoxiaopeng/wangjue/VQCell/FoldToken2')
sys.path.append('/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4')
sys.path.append('/guoxiaopeng/wangjue/VQCell/FoldToken2/scGCOT')
from scGCOT.utils.util_print import cal_cluster_metrics
from FoldToken4.main import create_parser
from FoldToken4.model_interface import MInterface
from sklearn.cluster import KMeans
from torch_scatter import scatter_mean
import numpy as np
import pandas as pd
import json


def get_topk_genes(data, query, k=20):
    """Get top k genes based on variance."""
    indexes = list(data0.obs['cell_type'] == query)
    data = data[indexes]
    counts = torch.bincount(data.flatten())
    values, indices = torch.topk(counts, k)
    topk_genes = [gene_list[i] for i in indices.tolist()]
    importance = values / values.max()
    return topk_genes, importance

gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])
data_path = '/guoxiaopeng/wangjue/data/celldata/scGPT/data/cellxgene/test_data/pancreas_sub.h5ad'
data0 = sc.read(data_path)
vqs, geneexpemb, indexes = torch.load('./results/pancrea_vq.pth')
indexes = torch.from_numpy(np.concatenate(indexes, axis=0))
cell_uni = data0.obs['cell_type'].unique().tolist()
acinar_genes, acinar_importance = get_topk_genes(indexes, 'acinar cell', k=100)
endocrine_genes, endocrine_importance = get_topk_genes(indexes, 'endocrine cell', k=100)
pancreatic_genes, pancreatic_importance = get_topk_genes(indexes, 'pancreatic ductal cell', k=100)
myeloid_genes, myeloid_importance = get_topk_genes(indexes, 'myeloid cell', k=100)
stromal_genes, stromal_importance = get_topk_genes(indexes, 'stromal cell', k=100)
glial_genes, glial_importance = get_topk_genes(indexes, 'glial cell', k=100)
enteric_genes, enteric_importance = get_topk_genes(indexes, 'enteric neuron', k=100)
mesothelial_genes, mesothelial_importance = get_topk_genes(indexes, 'mesothelial cell', k=100)

res = {'acinar': {'genes': acinar_genes, 'importance': acinar_importance.tolist()},
       'endocrine': {'genes': endocrine_genes, 'importance': endocrine_importance.tolist()},
       'pancreatic ductal': {'genes': pancreatic_genes, 'importance': pancreatic_importance.tolist()},
       'myeloid': {'genes': myeloid_genes, 'importance': myeloid_importance.tolist()},
       'stromal': {'genes': stromal_genes, 'importance': stromal_importance.tolist()},
       'glial': {'genes': glial_genes, 'importance': glial_importance.tolist()},
       'enteric neuron': {'genes': enteric_genes, 'importance': enteric_importance.tolist()},
       'mesothelial': {'genes': mesothelial_genes, 'importance': mesothelial_importance.tolist()}}
       }
w = open('./results/pancreas_topk_genes.json', 'w')
w.write(json.dumps(res, indent=4))

# sc_data = sc.AnnData(data0.X.toarray())
# sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
# sc.tl.umap(sc_data)
# sc_data.obs['true_labels'] = data0.obs['cell_type'].tolist()
# sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_raw', frameon=True)

# sc_data = sc.AnnData(geneexpemb.numpy())
# sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
# sc.tl.umap(sc_data)
# sc_data.obs['true_labels'] = data0.obs['cell_type'].tolist()
# sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_chriscell', frameon=True)
