import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy as c

# 示例数据
labels = ['RPL10A', 'RPS7', 'RPL17', 'CD63', 'RPLP0', 'RPL11', 'SET', 'PNN', 'ASPM', 'RPL38', 'TAF11', 'RPL9', 'RPS13']

data1 = [3.2485, 3.6125, 3.1617, 1.9484, 3.7765, 3.7270, 2.6923, 1.6247, 0.8145,
        2.1813, 0.6690, 3.3543, 3.1000]

data2 = [3.2305, 3.5802, 3.1664, 1.9452, 3.7699, 3.6809, 2.6757, 1.6344, 0.7749,
        2.2273, 0.6988, 3.2657, 3.1223]

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
plt.ylabel('Expression value', fontsize=18)

# 设置 X 轴的刻度
plt.xticks(x, labels, fontsize=15, rotation=90)

# 添加图例
plt.legend(fontsize = 14)

# 显示图形
plt.tight_layout()
plt.savefig('visualize/figures/aacs_ev.pdf')