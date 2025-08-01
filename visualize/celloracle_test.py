import os
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import celloracle as co

adata = sc.read('/guoxiaopeng/wangjue/data/celldata/scGPT_processed_datasets/Fig3_Perturbation/Fig3_AB_PerturbPred/k562_1900_100_re_ctrl_sample/perturb_processed.h5ad')[:100]
base_GRN = co.data.load_human_promoter_base_GRN()
oracle = co.Oracle()
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.normalize_total(adata, target_sum=1e4)  # 标准化
sc.pp.log1p(adata)  # 对数变换
sc.pp.highly_variable_genes(adata, n_top_genes=2000)  # 选择高变基因
adata = adata[:, adata.var['highly_variable']]  # 筛选高变基因

sc.pp.neighbors(adata, n_neighbors=10)
sc.tl.louvain(adata, resolution=1.0)
oracle.import_anndata_as_raw_count(adata=adata,
                                   cluster_column_name="louvain",
                                   embedding_name="X_pca")
oracle.import_TF_data(TF_info_matrix=base_GRN)
oracle.perform_PCA()
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
k=10
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                      b_maxl=k*4, n_jobs=4)
links = oracle.get_links(cluster_name_for_GRN_unit="louvain", alpha=10,
                         verbose_level=10)
links.filter_links()
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10,
                              use_cluster_specific_TFdict=True)
goi = "AACS"
# sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
# plt.savefig("AACS_hist.png")
oracle.simulate_shift(perturb_condition={goi: 1.0},
                      n_propagation=3)
# Get transition probability
oracle.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle.calculate_embedding_shift(sigma_corr=0.05)