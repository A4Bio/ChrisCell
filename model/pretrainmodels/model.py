# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited

import torch
import torch.nn.functional as F
from torch import nn
from modules.vq_modules import SoftCVQLayer

def exists(val):
    return val is not None

class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id = None, pad_token_id = None):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        # print('self.bin_num_idx',self.bin_num_idx, self.bin_num_idx.shape)

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x==self.mask_token_id).nonzero()
        x_pad_idx = (x==self.pad_token_id).nonzero()
        # print("x_mask",x_mask_idx.shape,x_mask_idx)
        
        x = self.mlp(x) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x) # [B, N, H]
        # print('weight', weight.shape, weight, torch.sum(weight, 2))
        
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        # print('bin_num_idx', bin_num_idx.shape)
        
        token_emb = self.emb(bin_num_idx) # [H, D]
        # print('token_emb', token_emb.shape)
        x = torch.matmul(weight, token_emb) #[B, N, D]
    
        # print("x_emb",x.shape,x)
        
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        # print(mask_token_emb.dtype)
        # print("x", x.dtype)
        x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)
        # print("x_emb",x.shape,x)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x

class RandomPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class Model(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,  # num of tokens
            max_seq_len,  # max length of sequence
            embed_dim,  # encoder dim of tokens
            decoder_embed_dim,
            tie_embed=False,
            bin_alpha = 1.0,
            bin_num = 10,
            pad_token_id = None,
            mask_token_id = None,
            level=12,
            condition_layer=6,
            latent_dim=32,
            celltype_num=831,  # number of cell types
            tissue_num=377,  # number of tissue types
            disease_num = 150
    ):
        super(Model, self).__init__()

        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        # encoder
        self.token_emb = AutoDiscretizationEmbedding2(embed_dim, max_seq_len, bin_num=bin_num, bin_alpha=bin_alpha, pad_token_id=self.pad_token_id, mask_token_id=self.mask_token_id)
        self.pos_emb = nn.Embedding(max_seq_len+1, embed_dim)  #RandomPositionalEmbedding(embed_dim, max_seq_len)

        # ## DEBUG
        self.encoder = None
        # self.gene_vq = HierCVQLayer(level, embed_dim, latent_dim, levels=[level], condition_layer=condition_layer, num_codebooks=19266)
        # self.encoder_performer = Performer(dim=embed_dim, depth=2, heads=8, dim_head=embed_dim//8)
        print('level:', level, 'latent_dim:', latent_dim, 'condition_layer:', condition_layer)
        self.cell_vq = SoftCVQLayer(level, embed_dim, latent_dim, condition_layer=condition_layer)

        ##### decoder
        self.decoder = None
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)

        # prediction head
        self.cell_type = nn.Linear(4*embed_dim, celltype_num)  
        self.tissue = nn.Linear(4*embed_dim, tissue_num)  
        self.disease = nn.Linear(4*embed_dim, disease_num)  # for disease prediction
        self.sex = nn.Linear(4*embed_dim, 3)
        self.age = nn.Linear(4*embed_dim, 184)
        self.mean = nn.Linear(decoder_embed_dim, 1)
        self.disp = nn.Linear(decoder_embed_dim, 1)
        self.pi = nn.Linear(decoder_embed_dim, 1)

        # go
        self.go = torch.eye(19266)
        self.go = nn.Parameter(self.go, requires_grad=True)
        self.go_embed = nn.Linear(19266, embed_dim)
    

    def get_cellemb(self, x, 
                padding_label, 
                encoder_position_gene_ids, 
                output_attentions=False,
                **kwargs):
        geneemb = self.encode(x, padding_label, encoder_position_gene_ids, output_attentions=output_attentions, **kwargs)
        geneemb1 = geneemb[:,-1,:]
        geneemb2 = geneemb[:,-2,:]
        geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
        geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)
        geneembmerge = torch.concat([geneemb1,geneemb2,geneemb3,geneemb4],axis=1)
        geneembmax, _ = torch.max(geneemb, dim=1)
        return geneembmerge, geneembmax

    def get_cellcode(self, x, 
                padding_label, 
                encoder_position_gene_ids, 
                output_attentions=False, 
                **kwargs):
        geneemb = self.encode(x, padding_label, encoder_position_gene_ids, output_attentions=output_attentions, **kwargs)
        cellemb, indexes = torch.max(geneemb, dim=1)
        _, indexes2 = torch.max(geneemb[:, :-2], dim=1)
        _, cell_code, vq_loss = self.cell_vq(cellemb, temperature=1e-8, vqshortcut=False)
        return geneemb, cell_code, encoder_position_gene_ids[:,indexes2[0]]


    def encode(self, x, 
                padding_label, 
                encoder_position_gene_ids, 
                output_attentions=False, 
                **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        with torch.no_grad():
            # token and positional embedding
            x = self.token_emb(torch.unsqueeze(x, 2), output_weight = 0)
            if output_attentions:
                x.requires_grad_()  # used for attn_map output

            position_emb = self.pos_emb(encoder_position_gene_ids)
            go_emb = self.go_embed(self.go[encoder_position_gene_ids])

            x += position_emb
            x += go_emb
            x = self.encoder(x, padding_mask=padding_label)

        return x
    
    def get_properties(self, x, 
                padding_label, 
                encoder_position_gene_ids, 
                output_attentions=False, 
                **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        with torch.no_grad():
            # token and positional embedding
            x = self.token_emb(torch.unsqueeze(x, 2), output_weight = 0)
            if output_attentions:
                x.requires_grad_()  # used for attn_map output

            position_emb = self.pos_emb(encoder_position_gene_ids)
            go_emb = self.go_embed(self.go[encoder_position_gene_ids])

            x += position_emb
            x += go_emb
            x = self.encoder(x, padding_mask=padding_label)
            cellemb = self.get_cellemb(x, encoder_position_gene_ids, padding_label)

            cell_type = self.cell_type(cellemb) # [B,N,C]
            tissue = self.tissue(cellemb)  # [B,N,T]
            disease = self.disease(cellemb)
            sex = self.sex(cellemb)
            age = self.age(cellemb)
        return cell_type, tissue, disease, sex, age
