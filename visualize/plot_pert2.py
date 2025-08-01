import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy as c

#data = torch.load('visualize/3906.pth')
# 示例数据
labels = ['Linear', 'GEARS', 'scGPT', 'VQCell']
#data = data['values']
data = [0.13, 0.15, 0.22, 0.225]

# 设置柱的宽度
bar_width = 0.5
plt.figure(figsize=(8, 4))

# 设置 X 轴的位置
x = np.arange(len(labels))

# 创建柱形图
plt.bar(x, data, width=bar_width, color='skyblue')

# 添加标题和标签
plt.title('Perturbation profile prediction', fontsize=20)
plt.xlabel('Methods', fontsize=18)
plt.ylabel('Pearson coefficient', fontsize=18)

# 设置 X 轴的刻度
plt.xticks(x, labels, fontsize=18, rotation=90)

# 添加图例
plt.legend(fontsize = 14)

# 显示图形
plt.tight_layout()
plt.savefig('figures/pearson.pdf')