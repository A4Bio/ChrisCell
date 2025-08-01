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

cell_type_int = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/zheng/zheng-test-label.npy')
cluster_num = len(np.unique(cell_type_int)) - 1
print(cluster_num)

# data_path = '/guoxiaopeng/wangjue/data/celldata/scGPT/data/cellxgene/test_data/pancreas_sub.h5ad'
# data0 = sc.read(data_path)
# cell_uni = data0.obs['cell_type'].unique().tolist()
# cell_type_int = np.array([cell_uni.index(item) for item in data0.obs['cell_type']])
# cluster_num = 12
# print(cluster_num)

emb1, emb2 = torch.load('./results/maskgene_zinb_zheng4.pth')

emb1, emb2 = emb1.detach().cpu().numpy(), emb2.detach().cpu().numpy()

print(emb1.shape, emb2.shape)

kmeans = KMeans(cluster_num)
kmeans.fit(emb1)
acc1, nmi1, ari1, sil1 = cal_cluster_metrics(cell_type_int, kmeans.labels_, emb1)
print(acc1, nmi1, ari1, sil1)

kmeans = KMeans(cluster_num)
kmeans.fit(emb2)
acc2, nmi2, ari2, sil2 = cal_cluster_metrics(cell_type_int, kmeans.labels_, emb2)
print(acc2, nmi2, ari2, sil2)



# sc_data = sc.AnnData(emb1)
# sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
# sc.tl.umap(sc_data)
# sc_data.obs['true_labels'] = data0.obs['cell_type'].tolist()
# sc_data.obs['cluster_labels'] = ['cluster ' + str(label) for label in kmeans.labels_]
# sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_emb1_true', frameon=True)
# sc.pl.umap(sc_data,color=['cluster_labels'],wspace=0.3,size=30,save='pancreas_emb1_cluster', frameon=True)

# sc_data = sc.AnnData(emb2)
# sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
# sc.tl.umap(sc_data)
# sc_data.obs['true_labels'] = data0.obs['cell_type'].tolist()
# sc_data.obs['cluster_labels'] = ['cluster ' + str(label) for label in kmeans.labels_]
# sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_emb2_true', frameon=True)
# sc.pl.umap(sc_data,color=['cluster_labels'],wspace=0.3,size=30,save='pancreas_emb2_cluster', frameon=True)