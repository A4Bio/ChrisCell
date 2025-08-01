from src.modules.transformer_vanilla import TransformerBlock
import torch.nn as nn
import torch
from torch_scatter import scatter_mean

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout, extract_layers, dim_linear_block):
        super().__init__()
        self.layer = nn.ModuleList()
        self.extract_layers = extract_layers
        self.avg_pool = AvgPooling()
        self.block_list = nn.ModuleList()
        
        for _ in range(num_layers):
            self.block_list.append(
                TransformerBlock(dim=embed_dim, heads=num_heads, dim_linear_block=dim_linear_block, dropout=dropout,
                                 prenorm=False))

    def forward(self, x, mask=None):
        for depth, layer_block in enumerate(self.block_list):
            x = layer_block(x, mask)
            x = self.avg_pool(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, vq_dim, num_layers, dropout):
        super().__init__()
        self.layer = nn.ModuleList()
        self.block_list = nn.ModuleList()
        self.input = nn.Linear(vq_dim, embed_dim)
        for _ in range(num_layers):
            self.block_list.append(
                MLP(embed_dim, embed_dim*4, embed_dim, batch_norm=True, dropout=dropout))

    def forward(self, x):
        x = self.input(x)
        for depth, layer_block in enumerate(self.block_list):
            x = layer_block(x)
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim, vq_dim, num_layers, dropout):
        super().__init__()
        self.layer = nn.ModuleList()
        self.block_list = nn.ModuleList()
        self.out = nn.Linear(embed_dim, vq_dim)
        for _ in range(num_layers):
            self.block_list.append(
                MLP(embed_dim, embed_dim, embed_dim, batch_norm=True, dropout=dropout))

    def forward(self, x):
        for depth, layer_block in enumerate(self.block_list):
            x = layer_block(x)
        return self.out(x)
    
class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.LayerNorm(in_channels))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.LayerNorm(out_channels))
            else:
                module.append(nn.LayerNorm(mid_channels))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)


class AvgPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        seq = torch.arange(x.shape[1]).to(x.device)
        idx = torch.div(seq, 2, rounding_mode='floor')
        idx = torch.cat([idx, idx[-1].view((1,))])

        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)
        x = scatter_mean(x, index=idx, dim=1)
        return x