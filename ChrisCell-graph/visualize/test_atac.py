import scanpy as sc
import torch
import sys
from graph_module.main import create_parser
from train.model_interface import MInterface
import numpy as np

args = create_parser()
args.steps_per_epoch = 0
args.hidden_dim = 128
args.latent_dim = 32
args.layers = 1
args.vq_space = 12
args.levels = [12]
args.gene_num = 344592
args.topk_gene = 1000
model = MInterface(**vars(args))
model_path = '/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4/results/vqcell_v8_atac/checkpoints/model.pt'
params = torch.load(model_path)
params = {key.replace('_forward_module.',''):val for key, val in params.items()}
model.load_state_dict(params, strict=True)
model.cuda()
print('load pretrained model successfully')

data_path = '/guoxiaopeng/wangjue/data/celldata/simba_data/atac_seq_fig4.h5ad'
data0 = sc.read(data_path)
data = {'X':torch.from_numpy(data0.X.toarray()).cuda(), 'temp':1e-8}


vq_code, vq_emb, Cell_rep, h_V_emb, features, gene_indexes, attention = model.encode(data, None, 0, level=12, batch_size=128)

print(vq_code.unique().shape)
attention = attention.mean(1)

Cell_rep, vq_code, vq_emb, features, gene_indexes = map(lambda x:x.detach().cpu().numpy(), [Cell_rep, vq_code, vq_emb, features, gene_indexes])


sc_data = sc.AnnData(np.concatenate([features,features2], axis=0))
sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
sc.tl.umap(sc_data)
sc_data.obs['true_labels'] = ["Peaks"] * 20000 + list(map(str, range(100)))

sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=50,save='atacfeature', frameon=False)
