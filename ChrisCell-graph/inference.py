import scanpy as sc
import torch
import sys
sys.path.append('./train')
sys.path.append('./src')
from train.model_interface import MInterface
import argparse
import pandas as pd


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--res_dir', default='./train/results', type=str)
    parser.add_argument('--ex_name', default='', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--data_path', default='examples/brain_sub.h5ad')
    parser.add_argument('--level', default=12, type=int)
    parser.add_argument('--output_name', default='example', type=str)
    
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


def vq_decode(level, name, model, gene_idx = None):
    vq_code, vq_emb, Cell_rep, h_V_emb, features, gene_indexes, attention = model.encode(data, None, 0, level=level)
    Cell_rep, vq_code, vq_emb, features, gene_indexes = map(lambda x:x.detach().cpu().numpy(), [Cell_rep, vq_code, vq_emb, features, gene_indexes])

    torch.save([attention, vq_code, gene_indexes], 'brain.pth')

    # umap visualization based on Cell representation
    sc_data = sc.AnnData(Cell_rep)
    sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
    sc.tl.umap(sc_data)
    sc_data.obs['true_labels'] = data0.obs['cell_type'].tolist()
    sc_data.obs['pred_labels'] =  list(map(str, vq_code.tolist()))
    sc.pl.umap(sc_data,color=['true_labels'],wspace=0.3,size=30,save=name + '_celltype', frameon=True)
    sc.pl.umap(sc_data,color=['pred_labels'],wspace=0.3,size=30,save=name + '_cellstate', frameon=True)

    # umap visualization based on quantized embeddings
    sc_data = sc.AnnData(vq_emb)
    sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
    sc.tl.umap(sc_data)
    sc_data.obs['pred_labels'] =  list(map(str, vq_code.tolist()))
    sc.pl.umap(sc_data,color=['pred_labels'],wspace=0.3,size=30,save=name + '_vqemb', frameon=True)


if __name__ == '__main__':
    args = create_parser()
    args.steps_per_epoch = 0

    feature_path = 'feature.csv'
    genes = pd.read_csv(feature_path)['feature_name'].tolist()
    gene_idx = [i for i, gene in enumerate(genes) if 'RP' not in gene and gene != 'MALAT1']

    # load model
    model = MInterface(**vars(args))
    model_path = args.pretrained
    params = torch.load(model_path)
    params = {key.replace('_forward_module.','').replace('encoder1', 'encoder'):val for key, val in params.items()}
    model.load_state_dict(params, strict=False)
    model.to(args.device)
    print('load pretrained model successfully')

    # load data
    data_path = args.data_path
    data0 = sc.read(data_path)
    data = {'X':torch.from_numpy(data0.X.toarray()).cuda(), 'temp':1e-8, 'properties':None}
    # inference
    vq_decode(args.level, args.output_name, model, gene_idx=gene_idx)