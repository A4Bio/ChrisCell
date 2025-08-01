import scanpy as sc
import torch
import sys
sys.path.append('/storage/guoxiaopeng/wangjue/VQCell/FoldToken2')
sys.path.append('/storage/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4')
from FoldToken4.main import create_parser
from FoldToken4.model_interface import MInterface
import pickle
import numpy as np
import pandas as pd
from torch_scatter import scatter_sum, scatter_mean

args = create_parser()
args.steps_per_epoch = 0
args.hidden_dim = 128
args.latent_dim = 32
args.layers = 1
args.vq_space = 12
args.levels = [12]
args.gene_num = 6017
model = MInterface(**vars(args))
model_path = '/storage/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4/results/vqcell_v7_pert_nopert/checkpoints/model.pt'
params = torch.load(model_path)
params = {key.replace('_forward_module.',''):val for key, val in params.items()}
model.load_state_dict(params, strict=False)
model.cuda()
print('load pretrained model successfully')
gene_names = pd.read_csv('visualize/pert_gene.csv')['gene_name'].tolist()
#target_genes = ['RPL10A', 'RPS7', 'RPL17', 'CD63', 'RPLP0', 'RPL11', 'SET', 'PNN', 'ASPM', 'RPL38', 'TAF11', 'RPL9', 'RPS13']
target_genes = ['RPL10A', 'RPS7', 'RPL17', 'RPLP0', 'CD63', 'RPL11', 'SET', 'RPLP2', 'RPL36A', 'UBAC1']
tgt_indexes = [gene_names.index(name) for name in target_genes]

data_path = './visualize/sub_pert.pth'
data0 = torch.load(data_path)

data = {'X':[], 'temp':1e-8, 'Y':None, 'pert':[]}
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
    data['X'].append(y)
    data['pert'].append(z)
  #  data['pert'].append(pert)
    data['pert'].append(z)   
    Ys += ['Before perturbation', 'After perturbation']
    genes.append(item.pert)
data['X'] = torch.concat(data['X'], dim=0).cuda()
data['pert'] = torch.concat(data['pert'], dim=0).cuda()   

vq_code, vq_emb, Cell_rep, h_V_emb, features, gene_indexes, attention = model.encode(data, None, 0, level=12)
print(vq_code.unique().shape)

#torch.save([vq_code.reshape(-1, 2), genes], 'pert.pth')
print('save successfully')


attention = attention.mean(1)[:, 0, 1:]
# gene_indexes = gene_indexes[:200]
# attention = attention[:200].reshape(100, 2, -1)
# attention1, attention2 = attention[:,0], attention[:,1]
# gene_indexes1, gene_indexes2 = gene_indexes.reshape(100, 2, -1)[:, 0],  gene_indexes.reshape(100, 2, -1)[:, 1]

attention = scatter_sum(attention, gene_indexes, dim=-1)
# attention2 = scatter_sum(attention2, gene_indexes2, dim=-1).mean(0)
# values1, top_gene_indexes1 = torch.topk(attention1, k=10, dim=-1)
# values2, top_gene_indexes2 = torch.topk(attention2, k=10, dim=-1)

# genes1 = [gene_names[gene] for gene in top_gene_indexes1.tolist()]
# genes2 = [gene_names[gene] for gene in top_gene_indexes2.tolist()]
# save_data = {}
# save_data['before_genes'] = genes1
# save_data['after_genes'] = genes2
# save_data['values1'] = values1.tolist()
# save_data['values2'] = values2.tolist()
# torch.save(save_data, 'aacs_pert.pth')

vq_code_uni, vq_code_indexes = vq_code.unique(return_inverse=True)
code_weights = scatter_mean(attention, vq_code_indexes, dim=0)
values1, top_gene_indexes = torch.topk(code_weights, k=10, dim=-1)


Cell_rep, vq_code, vq_emb, features, gene_indexes = map(lambda x:x.detach().cpu().numpy(), [Cell_rep, vq_code, vq_emb, features, gene_indexes])

sc_data = sc.AnnData(Cell_rep)
#sc_data = sc.AnnData(data0.X.toarray())
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['pred_labels'] = vq_code
sc.pl.umap(sc_data,color=['pred_labels'],wspace=0.3,size=30,save='pert_nop_vqcode', frameon=True)
