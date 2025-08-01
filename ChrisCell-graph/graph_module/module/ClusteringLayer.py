import math
import torch
import torch.nn as nn
import torch.nn.init as init

class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, latent_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.randn(n_clusters, latent_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.clusters, a=math.sqrt(5))

    def forward(self, inputs):
        q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, dim=1) - self.clusters), dim=2) / self.alpha))
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        q = torch.transpose(torch.transpose(q, dim0=0, dim1=1) / torch.sum(q, dim=1), dim0=0, dim1=1)
        return q
