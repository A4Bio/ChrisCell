import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import numpy as np

vq_code, genes = torch.load('visualize/pert.pth')

def func(start, vq_code, genes):
    vq_code = vq_code.cpu()[start: start + 100]
    gene_name = genes[start].replace('+ctrl', '')
    vq_code_uni = vq_code.unique().tolist()
    cm = torch.zeros(size=[len(vq_code_uni), len(vq_code_uni)])
    for pair in vq_code:
        index1 = vq_code_uni.index(int(pair[0]))
        index2 = vq_code_uni.index(int(pair[1]))
        cm[index1, index2] += 1
    cm = cm.numpy()
    cm = np.round(cm / (cm.sum(-1)[:, None] + 1e-8), 2)

    # 设置绘图
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=vq_code_uni, yticklabels=vq_code_uni)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 添加标签和标题
    plt.xlabel('VQ Code', fontsize = 30)
    plt.ylabel('VQ Code', fontsize = 30)
    plt.title(gene_name + " gene perturbation", fontsize=40)
    plt.tight_layout()
    plt.savefig('visualize/figures/' + gene_name + '.pdf')

for i in range(100, 1000, 100):
    func(i, vq_code, genes)