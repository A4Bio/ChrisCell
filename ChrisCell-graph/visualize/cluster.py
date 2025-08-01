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



args = create_parser()
args.steps_per_epoch = 0
args.hidden_dim = 128
args.latent_dim = 32
args.layers = 1
args.vq_space = 12
args.levels = [12]
model = MInterface(**vars(args))
model_path = '/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4/results/vqcell_v7_alldata_mat_1l/checkpoints/model.pt'
params = torch.load(model_path)
params = {key.replace('_forward_module.',''):val for key, val in params.items()}
model.load_state_dict(params, strict=False)
model.cuda()
print('load pretrained model successfully')

data_path = '/guoxiaopeng/wangjue/data/celldata/scGPT/data/cellxgene/test_data/pancreas_sub.h5ad'
data0 = sc.read(data_path)
data = {'X':torch.from_numpy(data0.X.toarray()).cuda(), 'temp':1e-8}
cell_uni = data0.obs['cell_type'].unique().tolist()
cell_type_int = np.array([cell_uni.index(item) for item in data0.obs['cell_type']])

cluster_num = len(cell_uni)

# vq_code, vq_emb, Cell_rep, h_V_emb, features, gene_indexes, attention = model.encode(data, None, 0, level=12)
# vq_code_uni, uni_index = torch.unique(vq_code, return_inverse=True)
# vq_emb_uni = scatter_mean(vq_emb, uni_index, dim=0)
# print(vq_code.unique().shape)
# print(len(data0.obs['cell_type'].unique()))

# Cell_rep, vq_code, vq_emb, features, gene_indexes = map(lambda x:x.detach().cpu().numpy(), [Cell_rep, vq_code, vq_emb, features, gene_indexes])

# kmeans = KMeans(cluster_num)
# kmeans.fit(Cell_rep)
# acc, nmi, ari = cal_cluster_metrics(cell_type_int, kmeans.labels_)
# print(acc, nmi, ari)

# new_vq_code = kmeans.labels_[uni_index.cpu()]
# new_vq_emb = kmeans.cluster_centers_[kmeans.labels_][uni_index.cpu()]
# print('cluster finished')
a, b = np.load('kmeans.npy')

sc_data = sc.AnnData(data0.X.toarray())
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = data0.obs['cell_type'].tolist()
sc_data.obs['cluster_labels'] = ['cluster ' + str(label) for label in b]
sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save='pancreas_X_true', frameon=True)
sc.pl.umap(sc_data,color=['cluster_labels'],wspace=0.3,size=30,save='pancreas_X_cluster', frameon=True)
