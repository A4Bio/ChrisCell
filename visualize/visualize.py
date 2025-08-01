# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited

import argparse
import random,os
import numpy as np
import argparse
import torch
from tqdm import tqdm
import scanpy as sc
import torch.nn.functional as F
import sys
sys.path.append('/guoxiaopeng/wangjue/VQCell/VQCellV2')
from model.load import *
import h5py

####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('--local-rank', type=int, default=0, help='local_rank')
parser.add_argument('--task_name', type=str, default='deepcdr', help='task name')
parser.add_argument('--input_type', type=str, default='singlecell',choices=['singlecell','bulk'], help='input type; default: singlecell')
parser.add_argument('--output_type', type=str, default='gene',choices=['cell','gene','gene_batch','gene_expression'], help='cell or gene embedding; default: cell the difference between gene and gene_batch is that in gene mode the gene embedding will be processed one by one. while in gene_batch mode, the gene embedding will be processed in batch. GEARS use gene_batch mode.')
parser.add_argument('--pool_type', type=str, default='all',choices=['all','max'], help='pooling type of cell embedding; default: all only valid for output_type=cell')
parser.add_argument('--tgthighres', type=str, default='t4', help='the targeted high resolution (start with t) or the fold change of the high resolution (start with f), or the addtion (start with a) of the high resoultion. only valid for input_type=singlecell')
parser.add_argument('--data_path', type=str, default='./examples/cluster_19264.h5ad', help='input data path')
parser.add_argument('--save_path', type=str, default='./', help='save path')
parser.add_argument('--pre_normalized', type=str, default='F',choices=['F','T','A'], help='if normalized before input; default: False (F). choice: True(T), Append(A) When input_type=bulk: pre_normalized=T means log10(sum of gene expression). pre_normalized=F means sum of gene expression without normalization. When input_type=singlecell: pre_normalized=T or F means gene expression is already normalized+log1p or not. pre_normalized=A means gene expression is normalized and log1p transformed. the total count is appended to the end of the gene expression matrix.')
parser.add_argument('--demo', action='store_true', default=False, help='if demo, only infer 10 samples')
parser.add_argument('--version',  type=str, default='ce', help='only valid for output_type=cell. For read depth enhancemnet, version=rde For others, version=ce')
parser.add_argument('--model_path',  type=str, default='None', help='pre-trained model path')
parser.add_argument('--ckpt_name',  type=str, default='01B-resolution', help='checkpoint name')



args = parser.parse_args()

def inference(path, device, pretrainmodel, pretrainconfig):

    if path.endswith('.h5'):
        h5_file = h5py.File(path, 'r+')
        gexpr_feature = h5_file['X'][:]
    else:
        gexpr_feature = sc.read(path).X.toarray() if path.endswith('.h5ad') else np.load(path)
    genevqs=[]
    geneexpemb=[]
    cellembs = []
    indexes = []
    for i in tqdm(range(gexpr_feature.shape[0])):
        with torch.no_grad():
            #Bulk
            if args.input_type == 'bulk':
                if args.pre_normalized == 'T':
                    totalcount = gexpr_feature[i,:].sum()
                elif args.pre_normalized == 'F':
                    totalcount = np.log10(gexpr_feature[i,:].sum())
                else:
                    raise ValueError('pre_normalized must be T or F')
                tmpdata = (gexpr_feature[i,:]).tolist()
                pretrain_gene_x = torch.tensor(tmpdata+[totalcount,totalcount]).unsqueeze(0).to(device)
                data_gene_ids = torch.arange(19266, device=device).repeat(pretrain_gene_x.shape[0], 1)
            
            #Single cell
            if args.input_type == 'singlecell':
                # pre-Normalization
                if args.pre_normalized == 'F':
                    tmpdata = (np.log1p(gexpr_feature[i,:]/(gexpr_feature[i,:].sum())*1e4)).tolist()
                elif args.pre_normalized == 'T':
                    tmpdata = (gexpr_feature[i,:]).tolist()
                elif args.pre_normalized == 'A':
                    tmpdata = (gexpr_feature[i,:-1]).tolist()
                else:
                    raise ValueError('pre_normalized must be T,F or A')

                if args.pre_normalized == 'A':
                    totalcount = gexpr_feature[i,-1]
                else:
                    totalcount = gexpr_feature[i,:].sum()

                # select resolution
                if args.tgthighres[0] == 'f':
                    pretrain_gene_x = torch.tensor(tmpdata+[np.log10(totalcount*float(args.tgthighres[1:])),np.log10(totalcount)]).unsqueeze(0).to(device)
                elif args.tgthighres[0] == 'a':
                    pretrain_gene_x = torch.tensor(tmpdata+[np.log10(totalcount)+float(args.tgthighres[1:]),np.log10(totalcount)]).unsqueeze(0).to(device)
                elif args.tgthighres[0] == 't':
                    pretrain_gene_x = torch.tensor(tmpdata+[float(args.tgthighres[1:]),np.log10(totalcount)]).unsqueeze(0).to(device)
                else:
                    raise ValueError('tgthighres must be start with f, a or t')
                data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

            value_labels = pretrain_gene_x > 0

            #Gene embedding
            if args.output_type=='gene':
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(pretrain_gene_x.float(),pretrain_gene_x.float(),pretrainconfig)
                cellemb, vq_codes, index = pretrainmodel.get_vqcode(x=encoder_data, 
                                                padding_label=encoder_data_padding, 
                                                encoder_position_gene_ids=encoder_position_gene_ids)
                
                genevqs.append(vq_codes.detach().cpu().numpy())
                geneexpemb.append(cellemb.detach().cpu().numpy())
                indexes.append(index.detach().cpu().numpy())
                
    genevqs = torch.from_numpy(np.squeeze(np.array(genevqs)))
    geneexpemb = torch.from_numpy(np.squeeze(np.array(geneexpemb)))
    torch.save([genevqs, geneexpemb, indexes], './results/pancrea_vq.pth')


def main():
    #Set random seed
    random.seed(0)
    np.random.seed(0)  # numpy random generator

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    local_rank=args.local_rank
    
    # Initialize the distribution
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:'+str(local_rank))

    #Load model
    if args.version == 'noversion':
        ckpt_path = args.model_path
        key=None
    else:
        ckpt_path = './model/models/models.ckpt'
        if args.output_type == 'cell':
            if args.version == 'ce':
                key = 'cell'
            elif args.version == 'rde':
                key = 'rde'
            else:
                raise ValueError('No version found')
        elif args.output_type == 'gene':
            key = 'gene'
        elif args.output_type == 'gene_batch':
            key = 'gene'
        elif args.output_type == 'gene_expression': # Not recommended
            key = 'gene'
        else:
            raise ValueError('output_mode must be one of cell gene, gene_batch, gene_expression')
    hparams = {'level': 14, 'condition_layer': 3, 'latent_dim': 32}
    pretrainmodel,pretrainconfig = load_model_frommmf(ckpt_path,'cell', device=device, params=hparams)
    params = torch.load('FoldToken4/results/vq_zinb/checkpoints/model.pt', map_location='cpu')['state_dict']
    params = {key.replace('model.',''):val for key, val in params.items()}
    pretrainmodel.load_state_dict(params)
    pretrainmodel.eval()

    file = args.data_path
    inference(file, device, pretrainmodel, pretrainconfig)

if __name__=='__main__':
    main()
