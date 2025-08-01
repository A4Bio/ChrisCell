from torch import nn

from .mhsa import MultiHeadSelfAttention, MultiHeadCrossAttention

class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dropout=0.1, activation=nn.GELU):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(dim=dim, heads=heads)
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim*4),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x, attention = self.mhsa(x)
        x = self.norm_1(self.drop(x) + x)
        x = self.norm_2(self.linear(x) + x)
        return x, attention


class TransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0, prenorm=False):
        super().__init__()
        self.block_list = [TransformerBlock(dim, heads, dim_head,
                                            dim_linear_block, dropout, prenorm=prenorm) for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, seq, mask=None):
        for layer in self.layers:
            x, attention = layer(x, seq, mask)
        return x

class CrossAttentionBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None, dropout=0.1,
                 mhsa=None, prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhca = MultiHeadCrossAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.mhca_s = MultiHeadCrossAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.drop_cross = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.norm_3 = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)

    def forward(self, x, seq, mask=None):
        x = self.norm(self.drop_cross(self.mhca(seq, x, x)) + x)
        seq = self.norm_s(self.drop_cross(self.mhca_s(x, seq, seq)) + seq)
        return x, seq