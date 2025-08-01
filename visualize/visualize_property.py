import scanpy as sc
import torch
import sys
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score

np.random.seed(0)


data_path = '/guoxiaopeng/wangjue/data/celldata/all_test_datasets/all_test_19264.h5ad'
data0 = sc.read(data_path)
emb, _ = torch.load('results/maskgene_zinb_all.pth')
emb = emb.numpy()
sex = data0.obs['sex']
age = data0.obs['development_stage']
tissue = data0.obs['tissue']


# sc_data = sc.AnnData(emb)
# sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
# sc.tl.umap(sc_data)
# sc_data.obs['true_labels'] = age.tolist()
# sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_emb_age', frameon=False)

# sc_data = sc.AnnData(emb)
# sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
# sc.tl.umap(sc_data)
# sc_data.obs['true_labels'] = sex.tolist()
# sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_emb_sex', frameon=False)

sc_data = sc.AnnData(emb)
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = tissue.tolist()
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=20,save='pancreas_emb_tissue', frameon=False)
