import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import torch

data_path = ''
data0 = sc.read(data_path)

attention, vq_code, gene_indexes = torch.load('atac.pth')
index342 = vq_code==342
index16 = vq_code==16
index560 = vq_code==560
index913 = vq_code==913

value342 = data0.X.toarray()[index342].mean(0)[500:700]
# 生成示例数据
np.random.seed(0)
cell_rank = np.arange(200)  # 细胞排名
probabilities = value342


plt.figure(figsize=(10, 5))

print(1)
# 绘制条形图
bars = plt.bar(cell_rank, probabilities, color='tomato', width=1.0)

# 去掉边框
for bar in bars:
    bar.set_edgecolor('none')  # 设置边框颜色为透明

# 设置坐标轴标签和标题
#plt.xlim(0, data0.shape[1])
plt.ylabel('Peaks', fontsize = 15)
plt.xlabel('Chr index', fontsize = 15)
plt.title('Peaks Distribution of State 342', fontsize=20)

# 添加水平线
plt.axhline(y=1e-3, color='r', linestyle='--', label='1 x 10^-3')
plt.legend()

# 显示图形
plt.savefig('visualize/figures/342_value.pdf')