# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited

import argparse
import random,os
import numpy as np
import argparse
import torch
from tqdm import tqdm
import scanpy as sc
import torch.nn.functional as F
from model.load import *

####################################Settings#################################
parser = argparse.ArgumentParser()
parser.add_argument('--input_type', type=str, default='singlecell',choices=['singlecell','bulk'], help='input type; default: singlecell')
parser.add_argument('--pool_type', type=str, default='all',choices=['all','max'], help='pooling type of cell embedding; default: all only valid for output_type=cell')
parser.add_argument('--data_path', type=str, default='./examples/cluster_19264.h5ad', help='input data path')
parser.add_argument('--save_path', type=str, default='./results/example_output.pth', help='save path')
parser.add_argument('--pre_normalized', type=bool, default=False, help='if normalized before input; default: False (F). choice: True(T), Append(A) When input_type=bulk: pre_normalized=T means log10(sum of gene expression). pre_normalized=F means sum of gene expression without normalization. When input_type=singlecell: pre_normalized=T or F means gene expression is already normalized+log1p or not. pre_normalized=A means gene expression is normalized and log1p transformed. the total count is appended to the end of the gene expression matrix.')
parser.add_argument('--mode',  type=str, default='m1')
parser.add_argument('--device',  type=str, default='cuda')
parser.add_argument('--model_path',  type=str, default='./model/models/models.ckpt', help='pre-trained model path')

args = parser.parse_args()

def get_cellemb(geneemb, pool_type='max'):
    geneemb1 = geneemb[:,-1,:]
    geneemb2 = geneemb[:,-2,:]
    geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
    geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)
    if pool_type=='all':
        geneembmerge = torch.concat([geneemb1,geneemb2,geneemb3,geneemb4],axis=1)
    elif pool_type=='max':
        geneembmerge, _ = torch.max(geneemb, dim=1)
    else:
        raise ValueError('pool_type must be all or max')
    return geneembmerge

def inference(path, device, pretrainmodel, config):

    gexpr_feature = sc.read(path).X.toarray() if path.endswith('.h5ad') else np.load(path)
    cellembs=[]
    cell_codes=[]
    for i in tqdm(range(gexpr_feature.shape[0])):
        with torch.no_grad():
            #Bulk
            if args.input_type == 'bulk':
                if args.pre_normalized:
                    totalcount = gexpr_feature[i,:].sum()
                else:
                    totalcount = np.log10(gexpr_feature[i,:].sum())
                tmpdata = (gexpr_feature[i,:]).tolist()
                pretrain_gene_x = torch.tensor(tmpdata+[totalcount,totalcount]).unsqueeze(0).to(device)
            
            #Single cell
            if args.input_type == 'singlecell':
                # pre-Normalization
                if not args.pre_normalized:
                    tmpdata = (np.log1p(gexpr_feature[i,:]/(gexpr_feature[i,:].sum())*1e4)).tolist()
                else:
                    tmpdata = (gexpr_feature[i,:]).tolist()
                totalcount = gexpr_feature[i,:].sum()
                pretrain_gene_x = torch.tensor(tmpdata+[4.0,np.log10(totalcount)]).unsqueeze(0).to(device)

            encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(pretrain_gene_x.float(),pretrain_gene_x.float(),config)
            x, cell_code, _ = pretrainmodel.get_cellcode(x=encoder_data, 
                                            padding_label=encoder_data_padding, 
                                            encoder_position_gene_ids=encoder_position_gene_ids, 
                                            output_attentions=False)
            cellembs.append(get_cellemb(x, pool_type='all').detach().cpu().numpy())
            cell_codes.append(cell_code)
    cellembs = torch.from_numpy(np.squeeze(np.array(cellembs)))
    cell_codes = torch.from_numpy(np.squeeze(np.array(cell_codes)))
    torch.save([cellembs, cell_codes], args.save_path)

def main():
    #Set random seed
    random.seed(0)
    np.random.seed(0)  # numpy random generator

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device(args.device)

    ckpt_path = args.model_path
        
    pretrainmodel, config = load_model_frommmf(ckpt_path,args.mode, device=device)
    pretrainmodel.eval()

    file = args.data_path
    inference(file, device, pretrainmodel, config)

if __name__=='__main__':
    main()
