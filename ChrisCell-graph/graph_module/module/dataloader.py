import os
import sys

import networkx as nx
import torch
import torch_geometric.utils as g_utils

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from module.graph_function import *
from utils.preprocess import *


def Data(matrix, norm=True):
    if norm:
        adj = get_adj(matrix)[1]
    else:
        adj = get_adj(matrix)[0]
    G = nx.from_numpy_array(adj)
    data = g_utils.from_networkx(G)
    data.x = torch.tensor(matrix, dtype=torch.float)
    return data , adj



def HomoData(matrix, Adj):
    G = nx.from_numpy_array(Adj)
    data = g_utils.from_networkx(G)
    data.x = torch.tensor(matrix, dtype=torch.float)
    return data



def read_count(dataname, highly_genes=500):
    x, y = prepro(rootPath + '/data/' + dataname + '/data.h5')
    x = np.ceil(x).astype(np.float32)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = normalize(adata, copy=True, highly_genes=highly_genes, size_factors=True, normalize_input=True,
                      logtrans_input=True)
    count = adata.X
    return count, y, x


def read_count_h5(dataname, highly_genes=500):
    x, y = read_h5(rootPath + '/data/' + dataname + '/data.h5')
    x = np.ceil(x).astype(np.float32)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = normalize(adata, copy=True, highly_genes=highly_genes, size_factors=True, normalize_input=True,
                      logtrans_input=True)
    count = adata.X
    return count, y, x

