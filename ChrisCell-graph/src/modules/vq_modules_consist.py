import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch.autograd import Function
import numpy as np
import pdb

def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()


def round_ste_soft(z, states, temp):
    """Round with straight through gradients."""
    shape = torch.tensor(z.shape)
    z = z.reshape(torch.prod(shape[:-1]), -1)
    steps = torch.arange(states.min(), states.max()+1, device=z.device, dtype=z.dtype)
    states = states.to(z.device, dtype=z.dtype)
    d = (z[...,None] - steps[None,None])**2
    mask = (states[0].reshape(-1,1)<=steps.reshape(1,-1)) & (steps.reshape(1,-1)<=states[1].reshape(-1,1))
    d = d*mask + 1e8*(~mask)
    
    alpha = 1/temp
    d = (alpha-1)* d.detach() +  d
        
    A = F.softmax(-d/temp, dim=-1)
    
    L, d, _ = A.shape
    code = torch.multinomial(A.reshape(L*d, -1), 1).view(L,d)
    code_one_hot = F.one_hot(code, num_classes=A.size(-1))
    c = code_one_hot*(1-A) + (1-code_one_hot)*(-A)
    A2 = A + c.detach()
    z = z + torch.einsum('ldc, c->ld', A2, steps) - z.detach()
    z = z.reshape(*shape)
    return z


class Quantizer(nn.Module):
    """Quantizer."""

    def __init__(self, levels: list[list], embedding_dim, eps: float = 1e-3):
        super(Quantizer, self).__init__()
        self._levels = levels
        self._eps = eps
        self._levels_np = [np.asarray(one) for one in levels]
        self._basis = [ np.concatenate(([1], np.cumprod(one[:-1]))).astype(np.int64)  for one in self._levels_np ]
        
        self._implicit_codebook = []
        for i in range(len(self._levels_np)):
            codebook = self.indexes_to_codes(torch.arange(self.codebook_size[i], dtype=torch.int64), self._basis[i], self._levels_np[i])
            self._implicit_codebook.append(codebook)
        
        self.proj = nn.ModuleList([nn.Linear(embedding_dim, len(level)) for level in levels])
        
        self.proj_inv = nn.ModuleList([nn.Linear(len(level), embedding_dim) for level in levels])
        
        self.temp = nn.Parameter(torch.tensor([1.0]))

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return [len(one) for one in self._levels]

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return [np.prod(one) for one in self._levels]

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor, idx) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np[idx] - 1) * (1 - self._eps) / 2
        offset = np.where(self._levels_np[idx] % 2 == 1, 0.0, 0.5)
        shift = np.tan(offset / half_l)
        
        half_l = torch.tensor(half_l, dtype=z.dtype,device=z.device)
        shift = torch.tensor(shift, dtype=z.dtype, device=z.device)
        offset = torch.tensor(offset, dtype=z.dtype, device=z.device)
        h = torch.tanh(z + shift) * half_l - offset
        return h

    def quantize(self, z: torch.Tensor, idx) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        
        half_l = torch.tensor((self._levels_np[idx] - 1) * (1 - self._eps) / 2)
        offset = np.where(self._levels_np[idx] % 2 == 1, 0.0, 0.3)
        # offset = torch.tensor(np.tan(offset / half_l))
        # offset = 0.5
        states = torch.round(torch.stack([-half_l, half_l], dim=0)-offset)
        
        MIN = torch.tensor(1e-8, device=z.device)
        temp = torch.max(self.temp**2, MIN)
        
        quantized = round_ste_soft(self.bound(z, idx), states, temp)
        

        # Renormalize to [-1, 1].
        half_width = self._levels_np[idx] // 2
        return quantized / torch.tensor(half_width, dtype=z.dtype, device=z.device)

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * torch.tensor(half_width, dtype=zhat_normalized.dtype,
                                               device=zhat_normalized.device)) + torch.tensor(half_width,
                                                                                              dtype=zhat_normalized.dtype,
                                                                                              device=zhat_normalized.device)

    def _scale_and_shift_inverse(self, zhat, levels_np):
        half_width = levels_np // 2
        return (zhat - torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)) / torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)

    def codes_to_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * torch.tensor(self._basis, dtype=zhat.dtype, device=zhat.device)).sum(dim=-1).type(
            torch.int64)  # 修改此处为 torch.int64

    def indexes_to_codes(self, indices: torch.Tensor, basis, levels_np) -> torch.Tensor:
        """Inverse of `codes_to_indexes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = np.mod(
            np.floor_divide(indices.cpu().numpy(), basis), levels_np
        )
        return self._scale_and_shift_inverse(
            torch.tensor(codes_non_centered, dtype=indices.dtype, device=indices.device), levels_np)

    def forward(self, h_in, levels='mix'):
        q_mix = 0
        q = 0
        r = h_in
        vq_code = []
        if levels == 'mix':
            n_iters = len(self._levels)
        else:
            n_iters = levels
        
        p = torch.rand(h_in.shape[0], device = h_in.device)
        for i in range(n_iters):
            h_i = self.proj[i](r)
            z_i = self.quantize(h_i, i)
            # z_i = h_i
            q_i = self.proj_inv[i](z_i)
            r = r - q_i
            q += q_i
            vq_code.append(z_i)
            select = (i/n_iters<p) & (p<=(i+1)/n_iters)
            q_mix = q_mix + q*select[:,None]
        
        # if levels == 'mix':
        #     quantized = q_mix
        # else:
        #     quantized = q
        
        quantized = q
        vq_code = torch.cat(vq_code, dim=-1)
        vq_loss = (h_in-q_mix).norm(dim=-1).mean()
        return quantized, vq_code, vq_loss

class HierCVQLayer(nn.Module):
    def __init__(self, embedding_dim, vq_dim, levels, condition_layer=6, sphere=True):
        super(HierCVQLayer, self).__init__()
        self.init = True
        self.log2_num_embeddings = 8
        self.levels = levels
        # self.register_buffer('embedding', bool_vectors.float())
        self.sphere = sphere
        hidden_dim = 1024
        # self.temp = nn.Parameter(torch.tensor([0.3]))
        self.temp = nn.Parameter(torch.tensor([0.3]))

        self.embedding_mlp = nn.ModuleDict(
            {f"code{i}": self.build_condition_layer(self.log2_num_embeddings,hidden_dim,condition_layer,vq_dim) for i in levels}
        )
        
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)
        
        self.init=False
        self.MSE = nn.HuberLoss(reduction='none')

    def build_condition_layer(self,log2_num_embeddings,hidden_dim,condition_layer,vq_dim):
        layers = [nn.Linear(log2_num_embeddings, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        for _ in range(condition_layer - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, vq_dim))
        return nn.Sequential(*layers)

    def get_bvectors(self, level, num_embeddings):
        int_range = torch.arange(0, num_embeddings)
        bool_vectors = (int_range[:, None] & (1 << torch.arange(level-1, -1, -1))) > 0
        return bool_vectors.to(self.temp.dtype)
        
    def project(self, h): 
        h = self.proj(h)
        return h
    
    def project_inv(self, h):
        h = self.proj_inv(h)
        return h

    def embed_id(self, vq_id, num_embeddings):
        level = int(math.log2(num_embeddings))
        bvectors = self.get_bvectors(level, num_embeddings)
        bvectors = F.pad(bvectors, (self.log2_num_embeddings-level,0,0,0), value=-1).to(vq_id.device)

        embed = getattr(self.embedding_mlp, f'code{2**level}')(bvectors)


        if self.sphere:
            embed = self.normalize(embed) # spherical
            
        return self.proj_inv(embed[vq_id])
    
    def get_code(self,h, attn_mask = None, temperature = 1e-5):
        h = self.proj(h)

        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            h = self.normalize(h) # spherical
            embed = self.normalize(embed) # spherical
        
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,0])

        h_flat = h[attn_mask==1]
        A, _ = self.attention(h_flat, embed, temperature)
        vq_code = A.argmax(dim=-1)
        return vq_code
    
    def decimal2binary(self, vqids):
        return self.embedding[vqids]
    
    def binary2decimal(self, binary_vector):
        base = 2 ** torch.arange(binary_vector.size(-1) - 1, -1, -1, device=binary_vector.device)
        vqids = (binary_vector * base).long().sum(dim=-1)
        return vqids

    def attention(self, H, C, temperature=1):
        alpha = 1/(self.temp**2)
        distances = - 2 * (alpha-1)* (H@C.t()).detach() - 2 * H@C.t()
        # distances = - 2 * alpha* (H@C.t())#.detach()

        A = F.softmax(-distances, dim=1)
        vq_code = distances.argmin(dim=-1)
        return A,  vq_code

    def normalize(self, x):
        return x/(torch.norm(x, dim=-1, keepdim=True)+1e-6)
    
    def get_vq(self,h, attn_mask = None, temperature = 1e-5):
        h = self.proj(h)

        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            h = self.normalize(h) # spherical
            embed = self.normalize(embed) # spherical
        
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,0])

        h_flat = h[attn_mask==1]
        A, code = self.attention(h_flat, embed, temperature)
        h_vq = embed[code]
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_code = torch.zeros(h.shape[:2], device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code
        quantized = self.proj_inv(quantized)
        return vq_code, quantized
    
    def entropy_loss(self, P, Q):
        return -torch.sum(P * torch.log(Q))

    def mask_code(self, code):
        code_b = self.decimal2binary(code)
        index_tensor = torch.randint(0, self.log2_num_embeddings, (code.shape[0],), device=code.device)

        # index_tensor = torch.ones_like(index_tensor)*8
        def create_mask(index_tensor, K):
            N = len(index_tensor)
            mask = torch.zeros((N, K), dtype=torch.bool, device=index_tensor.device)
            for i, idx in enumerate(index_tensor):
                mask[i, :idx] = True
            return mask
        mask = create_mask(index_tensor, self.log2_num_embeddings)
        code_b = code_b*(~mask)
        
        code = self.binary2decimal(code_b)
        return code


    def sample_code(self, h, level, temperature, attn_mask, sphere=True, num_embeddings=256):
        bvectors = self.get_bvectors(level, num_embeddings)
        bvectors = F.pad(bvectors, (self.log2_num_embeddings-level,0,0,0), value=-1).to(h.device)

        embed = getattr(self.embedding_mlp, f'code{2**level}')(bvectors)


        if sphere:
            embed = self.normalize(embed) # spherical

        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,0])
        h_flat = h[attn_mask==1]
        
        A, _ = self.attention(h_flat, embed, temperature)

        code = torch.multinomial(A, 1).view(-1)
        # code = A.argmax(dim=-1).view(-1)
        code_one_hot = F.one_hot(code, num_classes=A.size(-1))
        c = code_one_hot*(1-A) + (1-code_one_hot)*(-A)
        A2 = A + c.detach()
        h_vq = h_flat + A2@embed - h_flat.detach()
        
        vq_loss = (h_vq-h).norm(dim=-1).mean()
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        return quantized, code, vq_loss, h, A


    def forward(self, h_in,  mode='train', temperature=1, level = 8):
        h = self.proj(h_in)
        h = self.normalize(h) # spherical
        num_embeddings = [0]+self.levels
        
        quantized = 0
        vq_loss = 0
        p = torch.rand(h.shape[0], device = h.device)
        # if mode == 'train':
        #     idx_all = len(self.levels)
        # else:
        #     idx_all = self.levels.index(level)+1
        
        idx_all = self.levels.index(level)+1
        
        probs = 0
        vq_code = 0
        for idx in range(idx_all):
            level = self.levels[idx]
            level = int(math.log2(level))
            quantized_, code, vq_loss_, embed, A = self.sample_code(h, level, temperature, None, sphere=True, num_embeddings=num_embeddings[idx+1]-num_embeddings[idx])
            
            select = (idx/idx_all<p) & (p<(idx+1)/idx_all)
            quantized += quantized_*select[:,None]
            vq_loss += vq_loss_
            
            idx_range = torch.arange(code.shape[0], device=code.device)
            # probs += A[idx_range, code]*select
            probs+= A*select[:,None]
            vq_code += code*select

        # quantized = quantized[0]*p[:,None] + quantized[1]*(1-p)[:,None]
        vq_loss = vq_loss/idx_all

        # vq_code = code
        quantized_inv = self.proj_inv(quantized)
        
        if mode=='train':
            vqshortcut = temperature
            N = h_in.shape[0]
            keep_idx = torch.randperm(N)[:int(1.0*vqshortcut*N)]
            replace = torch.zeros(*h_in.shape[:-1],1,device=h_in.device,dtype=h_in.dtype)
            replace[keep_idx] = 1
            quantized_inv = quantized_inv*(1-replace) + h_in*replace
        
        # quantized_inv = h_in
            
        return quantized_inv, vq_code, quantized, probs, vq_loss




class FSQ(nn.Module):
    """Quantizer."""

    def __init__(self, levels: list[int], embedding_dim, vq_dim, eps: float = 1e-3):
        super(FSQ, self).__init__()
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(([1], np.cumprod(self._levels_np[:-1]))).astype(np.int64)  # 修改此处为 np.int64
        self._implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size, dtype=torch.int64))
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return np.prod(self._levels)

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = np.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = np.tan(offset / half_l)
        return torch.tanh(z + torch.tensor(shift, dtype=z.dtype, device=z.device)) * torch.tensor(half_l, dtype=z.dtype, device=z.device) - torch.tensor(offset, dtype=z.dtype, device=z.device)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = self._levels_np // 2
        return quantized / torch.tensor(half_width, dtype=z.dtype, device=z.device)

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * torch.tensor(half_width, dtype=zhat_normalized.dtype, device=zhat_normalized.device)) + torch.tensor(half_width, dtype=zhat_normalized.dtype, device=zhat_normalized.device)

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)) / torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)

    def codes_to_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * torch.tensor(self._basis, dtype=zhat.dtype, device=zhat.device)).sum(dim=-1).type(torch.int64)  # 修改此处为 torch.int64

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `codes_to_indexes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = np.mod(
            np.floor_divide(indices.cpu().numpy(), self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(torch.tensor(codes_non_centered, dtype=indices.dtype, device=indices.device))
    
    def forward(self, h_in, temperature=0, mode='train'):
        h = self.proj(h_in)
        quantized = self.quantize(h)
        vq_code = self.codes_to_indexes(quantized)
        quantized = self.proj_inv(quantized)
        return quantized, vq_code, 0
    
 