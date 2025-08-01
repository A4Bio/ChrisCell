import torch
from torch import nn
import torch.nn.functional as F

from module.ZINBLoss import ZINBLoss

Train_weight = [0, 0, 1, 0]
Finetune_weight =  [0.3, 0, 0, 1.5, 2]

def normalize_per_cell(expression_matrix, scaling_factor=1e4):
    # 计算每个细胞的总表达量
    cell_sums = expression_matrix.sum(dim=1, keepdim=True)
    
    # 避免除以零
    normalized_matrix = expression_matrix / (cell_sums + 1e-8) * scaling_factor

    return normalized_matrix

def scale_expression(expression_matrix):
    # 计算每个基因的均值和标准差
    mean = expression_matrix.mean(dim=0, keepdim=True)
    std = expression_matrix.std(dim=0, unbiased=False, keepdim=True)

    # 避免除以零
    scaled_matrix = (expression_matrix - mean) / (std + 1e-8)

    return scaled_matrix

def normalize(X):
    return torch.log1p(normalize_per_cell(X))

def contrastive_loss(pred, true, logit_scale):
    # normalized features
    pred = pred / pred.norm(dim=1, keepdim=True)
    true = true / true.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = logit_scale.exp()
    logits = logit_scale * pred @ true.t()
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(pred.device))

    return loss

def Train_loss(count, Mat, pi, disp, mean):

    Mat = Mat
    Mat_true = count
 #   Mat_loss =  contrastive_loss(Mat, Mat_true, logit_scale)
    Mat_loss = F.mse_loss(Mat, Mat_true)
    zinb = ZINBLoss()
    zinbloss = zinb(count, mean, disp, pi)
    loss = Mat_loss + zinbloss
    result = [Mat_loss, zinbloss, loss]
    return result



