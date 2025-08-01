import scanpy as sc
import torch
import sys
sys.path.append('/storage/guoxiaopeng/wangjue/VQCell/FoldToken2')
sys.path.append('/storage/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4')
from FoldToken4.main import create_parser
from FoldToken4.model_interface import MInterface
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from dmtev import DMTEV


args = create_parser()
args.steps_per_epoch = 0
args.hidden_dim = 128
args.latent_dim = 32
args.layers = 1
args.vq_space = 12
args.levels = [12]
model = MInterface(**vars(args))
model_path = '/storage/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4/results/vqcell_v7_alldata_mat_1l/checkpoints/model.pt'
params = torch.load(model_path)
params = {key.replace('_forward_module.',''):val for key, val in params.items()}
model.load_state_dict(params, strict=False)
model.cuda()
print('load pretrained model successfully')

data_path = '/storage/guoxiaopeng/wangjue/data/celldata/scGPT/data/cellxgene/test_data/brain_sub.h5ad'
data0 = sc.read(data_path)
data = {'X':torch.from_numpy(data0.X.toarray()).cuda(), 'temp':1e-8}

cluster_num = len(data0.obs['cell_type'].unique())
cell_types_uni = data0.obs['cell_type'].unique().tolist()
cell_types = [cell_types_uni.index(cell) for cell in data0.obs['cell_type'].tolist()]


def func(level, name):
    vq_code, vq_emb, Cell_rep, h_V_emb, features, gene_indexes, attention = model.encode(data, None, 0, level=level)
    # vq_code_uni, uni_index = torch.unique(vq_code, return_inverse=True)
    # vq_emb_uni = scatter_mean(vq_emb, uni_index, dim=0)
    print(vq_code.unique().shape)
    print(len(data0.obs['cell_type'].unique()))

    dmt = DMTEV(num_fea_aim=1, device_id=0, epochs=1500)
    X_dmt = dmt.fit_transform(Cell_rep.cpu())    
    
    Cell_rep, vq_code, vq_emb, features, gene_indexes = map(lambda x:x.detach().cpu(), [Cell_rep, vq_code, vq_emb, features, gene_indexes])

    # Plot the result
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_dmt[:, 0], X_dmt[:, 1], c=vq_code.tolist(), cmap='viridis', s=8)

    # Create legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)  # Add the legend to the current axes

    plt.title('DMT-EV visualization of VQCode')
    plt.xlabel('DMT-EV Component 1')
    plt.ylabel('DMT-EV Component 2')
    plt.savefig('dmt_vqcode.png')

func(12, str(12))