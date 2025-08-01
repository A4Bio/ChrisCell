import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv
from src.modules.vq_modules import HierCVQLayer
from graph_module.transformer_vanilla.transformer_block import TransformerBlock



class CellGraph(nn.Module):
    def __init__(self, gene_num, layers=2, hidden_dim=128, latent_dim=32, adj_dim=32, vq_levels=[], cell_communication=False):
        super(CellGraph, self).__init__()
        self.gene_num = gene_num
        self.transformer_blocks = nn.ModuleList()
        for i in range(layers):
            self.transformer_blocks.append(TransformerBlock(hidden_dim))
        self.vq = HierCVQLayer(vq_levels[-1], hidden_dim, latent_dim, levels=vq_levels, condition_layer=6)
        self.cell_communication = cell_communication
        if cell_communication:
            self.conv1 = TransformerConv(hidden_dim, hidden_dim)

    def forward(self, CellX, temp, CellEdgeIndex=None, level=12, vqshortcut=False, mode='train'):
        attention = None
        for transformer_block in self.transformer_blocks:
            CellX, attention = transformer_block(CellX)
            if self.cell_communication:
                Cell_rep, Cell_features = CellX[:, 0], CellX[:, 1:]
                Cell_rep = self.conv1(CellX, CellEdgeIndex)
                CellX = torch.cat([Cell_rep, Cell_features], dim=0)
            
        Cell_rep, Cell_features = CellX[:, 0], CellX[:, 1:]
        z_mean, vq_code, vq_loss, h_V_embed = self.vq(Cell_rep, temperature=temp, mode=mode, vqshortcut=vqshortcut, level=level)
        return  z_mean, vq_loss, vq_code, Cell_features, Cell_rep, h_V_embed, attention
