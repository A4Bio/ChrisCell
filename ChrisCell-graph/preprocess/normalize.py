#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2023 biomap.com, Inc. All Rights Reserved
# 
########################################################################

import scanpy as sc
import numpy as np
import scipy as sp
import torch

def normalize_per_cell(expression_matrix, scaling_factor=1e4):
    # 计算每个细胞的总表达量
    cell_sums = expression_matrix.sum(dim=1, keepdim=True)
    
    # 避免除以零
    normalized_matrix = expression_matrix / (cell_sums + 1e-8) * scaling_factor

    return normalized_matrix

def scale_expression(expression_matrix):
    # 计算每个基因的均值和标准差
    mean = expression_matrix.mean(dim=0, keepdim=True)
    std = expression_matrix.std(dim=0, unbiased=False, keepdim=True)

    # 避免除以零
    scaled_matrix = (expression_matrix - mean) / (std + 1e-8)

    return scaled_matrix

def highly_variable_genes(expression_matrix, n_top_genes=4000):
    # 计算平均表达
    mean_expression = expression_matrix.mean(dim=0)

    # 计算基因的方差
    variance_expression = expression_matrix.var(dim=0, unbiased=False)

    # 计算变异系数（CV）
    cv = variance_expression / (mean_expression + 1e-8)  # 防止除以零

    # 获取高变基因的索引
    _, top_indices = torch.topk(cv, n_top_genes)

    return top_indices


def normalize_torch(adata, size_factors=True):
    if isinstance(adata, sc.AnnData):
        adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error
    
    Xs = []
    adata.X = adata.X.toarray()
    for i in range(0, adata.X.shape[0], 20000):
        X = torch.from_numpy(adata.X[i: i+20000]).cuda()
        X = scale_expression(torch.log1p(normalize_per_cell(X, size_factors)))
        Xs.append(X.cpu().numpy())
    return torch.cat(Xs, dim=0).cpu().numpy()
    
    

def normalize(adata, copy=True, highly_genes=None, filter_min_counts=True, size_factors=True, normalize_input=True,
              logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    # if filter_min_counts:
    #     sc.pp.filter_genes(adata, min_counts=1)
    #     sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes,
                                    subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata
