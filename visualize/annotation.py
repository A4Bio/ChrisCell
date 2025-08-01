import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score
import torch
import scanpy as sc

# # 加载数据
# _, X_train = torch.load('results/maskgene_finetune_zhengtrain.pth')
# X_train = X_train.numpy()
# y_train = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/zheng/zheng-train-label.npy')
# _, X_test = torch.load('results/maskgene_finetune_zheng.pth')
# X_test = X_test.numpy()
# y_test = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/zheng/zheng-test-label.npy')
# print(X_train.shape)

# # 加载数据
# _, X_train = torch.load('results/scf_zheng_train.pth')
# X_train = X_train.numpy()
# y_train = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/zheng/zheng-train-label.npy')
# _, X_test = torch.load('results/zheng_scf.pth')
# X_test = X_test.numpy()
# y_test = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/zheng/zheng-test-label.npy')
# print(X_train.shape)

# # 加载数据
# _, X_train = torch.load('results/scf_seg_train.pth')
# X_train = X_train.numpy()
# y_train = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/Segerstolpe/Segerstolpe-train-label.npy')
# _, X_test = torch.load('results/scf_Seg.pth')
# X_test = X_test.numpy()
# y_test = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/Segerstolpe/Segerstolpe-test-label.npy')
# print(X_train.shape)

# # 加载数据
# _, X_train = torch.load('results/maskgene_finetune_Segtrain.pth')
# X_train = X_train.numpy()
# y_train = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/Segerstolpe/Segerstolpe-train-label.npy')
# _, X_test = torch.load('results/maskgene_finetune_Seg.pth')
# X_test = X_test.numpy()
# y_test = np.load('/guoxiaopeng/wangjue/data/celldata/scfoundation_data/cell_type_rawdata/Segerstolpe/Segerstolpe-test-label.npy')
# print(X_train.shape)

# data_path = '/guoxiaopeng/wangjue/data/celldata/scGPT/data/cellxgene/test_data/pancreas_sub.h5ad'
# #data_path = '/guoxiaopeng/wangjue/data/celldata/all_test_datasets/all_test_19264.h5ad'
# data0 = sc.read(data_path)
# X, _ = torch.load('results/scf.pth')
# y = data0.obs['cell_type']
# print(X.shape)
# # 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建模型
classifier = MLPClassifier(hidden_layer_sizes=(10,), max_iter=15, random_state=42)

# 训练模型
classifier.fit(X_train, y_train)

# 进行预测
y_pred = classifier.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
print(f"准确率: {accuracy:.4f}")
print(f"f1: {f1:.4f}")
print(f"recall: {recall:.4f}")
print(f"precison: {precision:.4f}")
print(classification_report(y_test, y_pred))