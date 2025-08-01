import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy as c

#data = torch.load('visualize/3906.pth')
# 示例数据
labels = ['RPL10A', 'RPS7', 'RPL17', 'RPLP0', 'CD63', 'RPL11', 'SET', 'RPLP2', 'RPL36A', 'UBAC1']
#data = data['values']
data = [2.5982, 2.6958, 2.5762, 2.7331, 2.0899, 2.7255, 2.4582, 2.5397, 2.4196,
        1.8245]

# 设置柱的宽度
bar_width = 0.5
plt.figure(figsize=(8, 4))

# 设置 X 轴的位置
x = np.arange(len(labels))

# 创建柱形图
plt.bar(x, data, width=bar_width, color='lightcoral')

# 添加标题和标签
plt.title('State 3906', fontsize=20)
plt.xlabel('Genes', fontsize=18)
plt.ylabel('Expression value', fontsize=18)

# 设置 X 轴的刻度
plt.xticks(x, labels, fontsize=15, rotation=90)

# 添加图例
plt.legend(fontsize = 14)

# 显示图形
plt.tight_layout()
plt.savefig('visualize/figures/3906_ev.pdf')