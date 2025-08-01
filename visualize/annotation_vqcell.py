import sys
sys.path.append('/guoxiaopeng/wangjue/VQCell/VQCellV2/')
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
import torch
import scanpy as sc
from model.load import *
import h5py

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

cell_types = h5py.File('/guoxiaopeng/wangjue/data/celldata/alltest.h5')
categories = sorted(list(set(cell_types['cell_type'])))
categories = list(map(lambda x:x.decode(), categories))
mask = np.ones([len(categories)],)

data_path = '/guoxiaopeng/wangjue/data/celldata/scGPT/data/cellxgene/test_data/pancreas_sub.h5ad'
data0 = sc.read(data_path)
y = [categories.index(item) for item in data0.obs['cell_type'].tolist()]
mask[list(set(y))] = 0



X, _ = torch.load('results/maskgene_finetune.pth')

print(X.shape)

ckpt_path = './model/models/models.ckpt'
hparams = {'level': 6, 'condition_layer': 3, 'latent_dim': 128}
pretrainmodel,pretrainconfig = load_model_frommmf(ckpt_path,'cell', device='cuda', params=hparams)
params = torch.load('FoldToken4/results/maskgene_finetune/checkpoints/model1.pt', map_location='cpu')
params = {key.replace('_forward_module.model.',''):val for key, val in params.items()}
pretrainmodel.load_state_dict(params)
pretrainmodel.eval()

y_pred = pretrainmodel.cell_type(X)


# 评估模型
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print(f"准确率: {accuracy:.2f}")
print(f"f1: {f1:.2f}")
print(f"recall: {recall:.2f}")
print(classification_report(y_test, y_pred))