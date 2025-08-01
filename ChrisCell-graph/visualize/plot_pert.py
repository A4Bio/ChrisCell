import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy as c

data = torch.load('visualize/3906.pth')
# 示例数据
labels = c(data['before_genes'] )
for gene in data['after_genes']:
    if gene not in labels:
        labels.append(gene)
data1 = []
data2 = []  # 第二组数据
for gene in labels:
    if gene in data['before_genes']:
        data1.append(data['values1'][data['before_genes'].index(gene)])
    else:
        data1.append(0)
    if gene in data['after_genes']:
        data2.append(data['values2'][data['after_genes'].index(gene)])
    else:
        data2.append(0)

# 设置柱的宽度
bar_width = 0.35
plt.figure(figsize=(8, 4))

# 设置 X 轴的位置
x = np.arange(len(labels))

# 创建柱形图
plt.bar(x - bar_width/2, data1, width=bar_width, label='Before perturbation', color='pink')
plt.bar(x + bar_width/2, data2, width=bar_width, label='After perturbation', color='lightblue')

# 添加标题和标签
plt.title('AACS', fontsize=20)
plt.xlabel('Genes', fontsize=18)
plt.ylabel('Attention Weight', fontsize=18)

# 设置 X 轴的刻度
plt.xticks(x, labels, fontsize=15, rotation=90)

# 添加图例
plt.legend(fontsize = 14)

# 显示图形
plt.tight_layout()
plt.savefig('visualize/figures/aacs_value.pdf')