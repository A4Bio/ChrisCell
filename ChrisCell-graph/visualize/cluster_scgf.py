import scanpy as sc
import torch
import sys
sys.path.append('/guoxiaopeng/wangjue/VQCell/FoldToken2')
sys.path.append('/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4')
sys.path.append('/guoxiaopeng/wangjue/VQCell/FoldToken2/scGCOT')
from scGCOT.utils.util_print import cal_cluster_metrics
from sklearn.cluster import KMeans
import numpy as np

data_path = '/guoxiaopeng/wangjue/data/celldata/scGPT/data/cellxgene/test_data/pancreas_sub.h5ad'
data0 = sc.read(data_path)
data = {'X':torch.from_numpy(data0.X.toarray()).cuda(), 'temp':1e-8}
cell_uni = data0.obs['cell_type'].unique().tolist()
cell_type_int = np.array([cell_uni.index(item) for item in data0.obs['cell_type']])

scf, scg = np.load('visualize/scfoundation.npy'), sc.read('visualize/scgpt.h5ad').X


cluster_num = 12


kmeans = KMeans(cluster_num)
kmeans.fit(scf)
acc1, nmi1, ari1 = cal_cluster_metrics(cell_type_int, kmeans.labels_)

kmeans1 = KMeans(cluster_num)
kmeans1.fit(scg)
acc2, nmi2, ari2 = cal_cluster_metrics(cell_type_int, kmeans1.labels_)

sc_data = sc.AnnData(scf)
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = data0.obs['cell_type'].tolist()
sc_data.obs['cluster_labels'] = ['cluster ' + str(label) for label in kmeans.labels_]
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_scf_true', frameon=True)
sc.pl.umap(sc_data,color=['cluster_labels'],wspace=0.3,size=30,save='pancreas_scf_cluster', frameon=True)

sc_data = sc.AnnData(scg)
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = data0.obs['cell_type'].tolist()
sc_data.obs['cluster_labels'] = ['cluster ' + str(label) for label in kmeans1.labels_]
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_scg_true', frameon=True)
sc.pl.umap(sc_data,color=['cluster_labels'],wspace=0.3,size=30,save='pancreas_scg_cluster', frameon=True)