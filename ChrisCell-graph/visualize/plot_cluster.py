import matplotlib.pyplot as plt
import numpy as np

# 示例数据
labels = ['NMI', 'ARI', 'ACC']  # 类别标签
group1 = [0.4621, 0.60716, 0.3875]          # 第一组数据
group2 = [0.3196, 0.417, 0.22222]          # 第二组数据
group3 = [0.2887, 0.35851, 0.02828]          # 第三组数据

# 设置柱宽
bar_width = 0.25

# 设置 x 轴位置
x = np.arange(len(labels))

# 创建柱形图
plt.bar(x - bar_width, group1, width=bar_width, label='VQCell rep', color = 'lightcoral')
plt.bar(x, group2, width=bar_width, label='VQCell VQcode', color='lightblue')
plt.bar(x + bar_width, group3, width=bar_width, label='Raw rep', color = 'lightgreen')

# 添加标签和标题
plt.ylabel('Value', fontsize=20)
plt.xticks(x, labels, fontsize=18)  # 设置 x 轴的刻度标签
plt.legend(fontsize=15)  # 添加图例

# 显示图形
plt.tight_layout()
plt.savefig('cluster.pdf')