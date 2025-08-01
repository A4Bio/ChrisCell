import os
import sys
import numpy as np
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils.utils import dopca

def get_adj(count, k=15, pca=50, mode="connectivity"):
    try:
        if pca:
            countp = dopca(count, dim=pca)
        else:
            countp = count
    except:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True, n_jobs=-1)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


