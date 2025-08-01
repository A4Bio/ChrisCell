import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_module.model.CellGraph import CellGraph

class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num=12, bin_alpha=1.0, mask_token_id = None, pad_token_id = 0.):
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
        
        self.emb_pad = nn.Embedding(1, self.dim)
        
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_pad_idx = (x==self.pad_token_id).nonzero()
        
        x = self.mlp(x[..., None]) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x) # [B, N, H]
        
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        
        token_emb = self.emb(bin_num_idx) # [H, D]
        x = torch.matmul(weight, token_emb) #[B, N, D]
        
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x


class HomoGraph(nn.Module):
    def __init__(self, gene_num, layers=2, hidden_dim=128, latent_dim=32, vq_levels = [12]):
        super(HomoGraph, self).__init__()
        self.gene_num = gene_num
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cell_emb = nn.Linear(gene_num, hidden_dim)
        self.gene_emb = nn.Embedding(gene_num, hidden_dim)
        self.expression_emb = AutoDiscretizationEmbedding2(hidden_dim, max_seq_len=200)
        self.out = nn.Linear(hidden_dim, gene_num)
        self.feature_emb = nn.Embedding(42384, hidden_dim, padding_idx=1)
        self.pos_emb = nn.Embedding(20, hidden_dim)
        self.encoder = CellGraph(self.gene_num, layers, hidden_dim, latent_dim, vq_levels=vq_levels)
        self.logits = nn.Linear(hidden_dim, 1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.temp = nn.Parameter(torch.ones([]))
        self.pi_decoder = nn.Linear(hidden_dim, gene_num)
        self.disp_decoder = nn.Linear(hidden_dim, gene_num)
        self.mean_decoder = nn.Linear(hidden_dim, gene_num)

    def forward(self, Cell, level=12, mode='train'):
        
        temp = self.temp
        Cell.x = self.cell_emb(Cell.x) 
        
        if hasattr(Cell, 'genes'):
            Cell.genes = self.gene_emb(Cell.gene_indexes) + self.expression_emb(Cell.genes)
            Cell.x = torch.cat([Cell.x[:, None], Cell.genes], dim = 1)
        if hasattr(Cell, 'properties'):
            Cell.properties = self.feature_emb(Cell.properties)
            Cell.x = torch.cat([Cell.x[:, None], Cell.properties], dim = 1)
        z_mean, vq_loss, vq_code, features, Cell_rep, h_V_emb, attention = self.encoder(Cell.x, temp, Cell.edge_index, level, mode='train')
        Matrix = self.out(z_mean)
        features_logits = self.logits(features)
        pi = torch.sigmoid(self.pi_decoder(z_mean))
        disp = torch.clamp(F.softplus(self.disp_decoder(z_mean)), min=1e-4, max=1e4)
        mean = torch.clamp(torch.exp(self.mean_decoder(z_mean)), min=1e-5, max=1e6)
        return  Matrix, vq_loss, vq_code, features_logits, self.temp, pi, disp, mean, Cell_rep

    def encode(self, Cell, temp=1.0, level=12):
        Cell.x = self.cell_emb(Cell.x) 
        if hasattr(Cell, 'genes'):
            Cell.genes = self.gene_emb(Cell.gene_indexes) + self.expression_emb(Cell.genes)
            Cell.x = torch.cat([Cell.x[:, None], Cell.genes], dim = 1)
        if hasattr(Cell, 'properties'):
            Cell.properties = self.feature_emb(Cell.properties)
            Cell.x = torch.cat([Cell.x[:, None], Cell.properties], dim = 1)
        z_mean, vq_loss, vq_code, features, Cell_rep, h_V_emb, attention = self.encoder(Cell.x, temp, Cell.edge_index, level, mode='valid')
        return vq_code, z_mean, Cell_rep, h_V_emb, features, attention



