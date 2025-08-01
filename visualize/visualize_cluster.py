import scanpy as sc
import torch
import sys
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score

np.random.seed(0)


cell_type_int = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/zheng/zheng-test-label.npy')
cell_type_str = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/zheng/zheng-str_label.npy', allow_pickle=True)
cell_types = [cell_type_str[i] for i in cell_type_int]
 
emb1 = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/zheng/data_test_count.npy')
emb2 = sc.read('/guoxiaopeng/wangjue/VQCell/scGPT/zheng_scg.h5ad').X
emb3, _ = torch.load('/guoxiaopeng/wangjue/VQCell/VQCellV2/results/zheng_scf.pth')
emb4, _ = torch.load('/guoxiaopeng/wangjue/VQCell/VQCellV2/results/maskgene_finetune_zheng.pth')
emb3, emb4 = emb3.numpy(), emb4.numpy()

print(emb1.shape, emb2.shape, emb3.shape, emb4.shape)

sc_data = sc.AnnData(emb1)
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = cell_types
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_emb1_true', frameon=False)

sc_data = sc.AnnData(emb2)
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = cell_types
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_emb2_true', frameon=False)

sc_data = sc.AnnData(emb3)
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = cell_types
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_emb3_true.png', frameon=False)

sc_data = sc.AnnData(emb4)
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = cell_types
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_emb4_true.png', frameon=False)