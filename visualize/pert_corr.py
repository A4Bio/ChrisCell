import scanpy as sc
import torch
import sys
sys.path.append('/guoxiaopeng/wangjue/VQCell/VQCell_private/train')
sys.path.append('/guoxiaopeng/wangjue/VQCell/VQCell_private/')
from train.model_interface import MInterface
import pickle
import numpy as np
import pandas as pd
import argparse
from torch_scatter import scatter_sum, scatter_mean
import matplotlib.pyplot as plt

import torch

def pearson_correlation_batch(x, y):
    # 确保输入为张量
    x = x.float()
    y = y.float()
    
    # 计算均值
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)
    
    # 计算差异
    diff_x = x - mean_x
    diff_y = y - mean_y
    
    # 计算 Pearson 相关系数
    numerator = torch.sum(diff_x * diff_y, dim=1)
    denominator = torch.sqrt(torch.sum(diff_x ** 2, dim=1) * torch.sum(diff_y ** 2, dim=1))
    
    # 避免除以零的情况
    denominator = torch.where(denominator == 0, torch.tensor(1.0), denominator)

    return numerator / denominator


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--res_dir', default='./train/results', type=str)
    parser.add_argument('--ex_name', default='', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--data_path', default='/guoxiaopeng/wangjue/data/celldata/celltype_specific_data/sub_pancreas.h5ad')
    parser.add_argument('--level', default=12, type=int)
    parser.add_argument('--output_name', default='pancreas', type=str)
    
    # Model parameters
    parser.add_argument('--topk_gene', default=200, type=int)
    parser.add_argument('--gene_num', default=60537, type=int)
    parser.add_argument('--vq_space', default=12, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--levels', default=[12], type=int, nargs='+')
    parser.add_argument('--condition_layer', default=6, type=int)
    parser.add_argument('--pretrained', default='pretrained_models/checkpoint.pt', type=str)
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--latent_dim', default=32, type=int)
    parser.add_argument('--cell_communication', default=False, type=str)

    args = parser.parse_args()
    return args

args = create_parser()
args.steps_per_epoch = 0
args.hidden_dim = 128
args.latent_dim = 32
args.layers = 1
args.vq_space = 12
args.levels = [12]
args.gene_num = 6017
model = MInterface(**vars(args))
model_path = '/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4/results/vqcell_v7_pert_nopert/checkpoints/model.pt'
params = torch.load(model_path)
params = {key.replace('_forward_module.','').replace('encoder1', 'encoder'):val for key, val in params.items()}
model.load_state_dict(params, strict=False)
model.cuda()
print('load pretrained model successfully')
gene_names = pd.read_csv('visualize/pert_gene.csv')['gene_name'].tolist()
# gene_names = [gene for i, gene in enumerate(gene_names) if 'RP' not in gene and gene != 'MALAT1']
# gene_idx = [i for i, gene in enumerate(gene_names) if 'RP' not in gene and gene != 'MALAT1']


data_path = './visualize/sub_pert.pth'
data0 = torch.load(data_path)

data = {'X':[], 'temp':1e-8, 'Y':[], 'pert':[]}
Xs = []
perts = []
Ys = []
genes = []
for item in data0:
    x = item.x[:, 0].unsqueeze(0)
    pert = item.x[:, 1].unsqueeze(0)
    y = item.y
    z = torch.zeros_like(x)
    data['X'].append(x)
  #  data['X'].append(x)
    data['Y'].append(y)
    data['pert'].append(z)
  #  data['pert'].append(pert)
    data['pert'].append(z)   
    genes.append(item.pert)
Ys = ['Before perturbation'] * len(data0) +  ['After perturbation'] * len(data0)
data['X'] = torch.concat(data['X'], dim=0).cuda()
data['Y'] = torch.concat(data['Y'], dim=0).cuda()
data['pert'] = torch.concat(data['pert'], dim=0).cuda()   
data['properties'] = None

aacs_index = 3857
diff = (data['Y'][:100] - data['X'][:100]).T
aacs = diff[aacs_index].unsqueeze(0).repeat(6017, 1)
corr = pearson_correlation_batch(aacs, diff)

diff_topk_v, diff_topk = torch.topk(torch.abs(diff), k=11, dim=-1)
corr_topkv, corr_topk = torch.topk(torch.abs(corr), k=11, dim=-1)
corr_topkv = corr_topkv[1:]
corr_topk = corr_topk[1:]
corr_genes = [gene_names[gene] for gene in corr_topk.tolist()]

# plt.figure(figsize=(8, 5))
# x = np.arange(len(corr_topk))
# plt.bar(x[:8], corr_topkv[:8].cpu().numpy(), label='Before perturbation', color='steelblue')
# plt.xticks(x[:8], corr_genes[:8], fontsize=15, rotation=90)
# #plt.tight_layout()
# plt.xlabel('Genes', fontsize=14)
# plt.ylabel('Coefficient', fontsize=18)
# plt.savefig('figures/aacs_corr.pdf',bbox_inches='tight')

new_gene_names = [gene_names[i] for i in corr_topk.tolist()]

vq_code, vq_emb, Cell_rep, h_V_emb, features, gene_indexes1, attention1 = model.encode(data, corr_topk.tolist(), 0, level=12)
data['X'] = data['Y']
vq_code, vq_emb, Cell_rep, h_V_emb, features, gene_indexes2, attention2 = model.encode(data, corr_topk.tolist(), 0, level=12)

attention1 = attention1.mean(1)[:, 0, 1:]
attention2 = attention2.mean(1)[:, 0, 1:]

attention1 = scatter_sum(attention1, gene_indexes1, dim=-1).mean(0)
attention2 = scatter_sum(attention2, gene_indexes2, dim=-1).mean(0)
diff_value, diff_index = torch.topk(torch.abs(attention1 - attention2), k=8, dim=-1)
diff_genes = [new_gene_names[i] for i in diff_index.tolist()]

plt.figure(figsize=(8, 5))
x = np.arange(8)
plt.bar(x[:8], diff_value.cpu().numpy(), label='Before perturbation', color='lightblue')
plt.xticks(x, diff_genes, fontsize=15, rotation=90)
#plt.tight_layout()
plt.xlabel('Genes', fontsize=14)
plt.ylabel('Attention weight difference', fontsize=18)
plt.savefig('figures/aacs_attn.pdf',bbox_inches='tight')

