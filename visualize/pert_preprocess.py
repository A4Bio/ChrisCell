import pandas as pd
import scanpy as sc


gene_list_df = pd.read_csv('./preprocess/features.csv', delimiter=',')
gene_list = list(gene_list_df['feature_name'])
gene_names = sc.read('/storage/guoxiaopeng/wangjue/data/celldata/scGPT_processed_datasets/Fig3_Perturbation/Fig3_AB_PerturbPred/k562_1900_100_re_ctrl_sample/perturb_processed.h5ad').var['gene_name'].tolist()

gene_indexes = []
gene_indexes1 = []
for i, gene_name in enumerate(gene_names):
    if gene_name not in gene_list:
        print(gene_name)
    else:
        gene_indexes.append(gene_list.index(gene_name))
        gene_indexes1.append(i)