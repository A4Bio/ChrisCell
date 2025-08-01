import h5py 
import scanpy as sc
import numpy as np
import pandas as pd
from multiprocessing import Pool
import os
from tqdm import tqdm

def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]

    return X_df

def preprocess(path, i):
    adata = sc.read(path)
    X = adata.X.toarray()
    col = adata.var['feature_name'].tolist()
    X_df = pd.DataFrame(X,index=list(range(X.shape[0])),columns=col)
    X_df = main_gene_selection(X_df, gene_list)

    w = h5py.File('/guoxiaopeng/wangjue/data/celldata/train_h5/partition_' + str(i) + '.h5', 'w')
    X = w.create_dataset('X', data=X_df.values, dtype=np.float32)
    print(i)
    w.close()
    del w


def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])

    path = '/guoxiaopeng/wangjue/data/celldata/scGPT/data/cellxgene/data/'
    files = list_all_files(path)

    for i, file in enumerate(tqdm(files)):
        try:
            if not os.path.exists('/guoxiaopeng/wangjue/data/celldata/train_h5/partition_' + str(i) + '.h5'):
                preprocess(file, i)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue


