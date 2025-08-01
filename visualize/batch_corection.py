import scanpy as sc
import torch
import sys
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score

np.random.seed(0)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    ind = np.array(list(zip(row_ind, col_ind)))
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def cal_cluster_metrics(y, y_pred, X):
    acc = np.round(cluster_acc(y, y_pred), 5)
    y = list(map(int, y))
    y_pred = np.array(y_pred)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    sil = silhouette_score(X, np.array(y))
    return acc, nmi, ari, sil

# cell_type_int = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/Segerstolpe/Segerstolpe-test-label.npy')
# cluster_num = len(np.unique(cell_type_int)) -1
# print(cluster_num)

# cell_type_int = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/zheng/zheng-test-label.npy')
# cluster_num = len(np.unique(cell_type_int)) - 1
# print(cluster_num)

#data_path = '/guoxiaopeng/wangjue/data/celldata/data/celldata/all_test_datasets/all_test_19264.h5ad'
data_path = '/guoxiaopeng/wangjue/data/celldata/all_test_datasets/all_test_19264.h5ad'
data0 = sc.read(data_path)
cell_uni = data0.obs['donor_id'].unique().tolist()
cell_type_int = np.array([cell_uni.index(item) for item in data0.obs['donor_id']])
print(len(cell_uni))

emb1, _ = torch.load('./results/maskgene_zinb_all.pth')
emb2, _ = torch.load('./results/scf_all.pth')
emb3 = sc.read('./results/scg_all.h5ad').X

emb1, emb2 = emb1.detach().cpu().numpy(), emb2.detach().cpu().numpy()


sil = silhouette_score(emb1, np.array(cell_type_int))
print(sil)

sil = silhouette_score(emb2, np.array(cell_type_int))
print(sil)

sil = silhouette_score(emb3, np.array(cell_type_int))
print(sil)

sil = silhouette_score(data0.X.toarray(), np.array(cell_type_int))
print(sil)

sc_data = sc.AnnData(emb1)
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = cell_type_int
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_vqcell_batch', frameon=True)

sc_data = sc.AnnData(emb2)
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = cell_type_int
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_scf_batch', frameon=True)

sc_data = sc.AnnData(emb3)
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = cell_type_int
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_scg_batch', frameon=True)

sc_data = sc.AnnData(data0.X.toarray())
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = cell_type_int
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_raw_batch', frameon=True)