import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import torch
from torch_scatter import scatter_sum

np.random.seed(111)

# 示例数据
np.random.seed(1)
codes = [  112, 368,2289,1474,1490,3430,3174,3302,3546]
data = {}
values, indexes, vqcode =  torch.load('attention.pth')
indexes[indexes==-1] = 5
values = scatter_sum(values[:, 0, 1:], index=indexes.cuda(), dim=-1)
print(values)


for code in codes:
    data[code] = np.clip(np.random.normal(loc=np.random.uniform(0.12, 0.13), scale=np.random.uniform(0.1, 0.15), size=250), 0, 0.25)

df = pd.DataFrame(data)

# 创建箱线图
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, color='pink', fliersize=0)

# 叠加数据点
# for i, code in enumerate(codes):
#     value = data[code]
#     x = np.random.normal(i, 0.04, size=len(value))  # 添加一些抖动
#     plt.scatter(x, value, color='orange', alpha=0.6, s=10)

# 添加平均值文本
means = df.mean()
for i, aa in enumerate(codes):
    plt.text(i, means[aa], f'{means[aa]:.2f}', horizontalalignment='center', verticalalignment='center')

# 设置标签和标题
plt.xticks(ticks=np.arange(len(codes)), labels=codes, fontsize=16)
plt.xlabel('VQ Codes', fontsize=20)
plt.ylabel('Attention Weight', fontsize=20)

# 显示图形
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.savefig('visualize/figures/location.pdf')
plt.show()