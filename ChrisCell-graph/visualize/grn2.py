import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import pandas as pd
import torch
from tqdm import tqdm
from torch_scatter import scatter_sum

gene_names = pd.read_csv('feature.csv')['feature_name'].tolist()
attention, vq_code, gene_indexes = torch.load('tcell_re.pth')
indexes = (vq_code == 552)
attention = attention[indexes][:, 0, 1:]
gene_indexes = gene_indexes[indexes]
attention = scatter_sum(attention.cpu(), dim=-1, index=torch.from_numpy(gene_indexes).cpu()).mean(0)
values, indexes = torch.topk(attention, k=10)
topk_genes = [gene_names[i] for i in indexes]

# 创建一个空的图
G = nx.Graph()
# 添加节点
nodes = []
# 添加边
edges = {}
    
G.add_nodes_from(nodes)
sorted_edges = sorted(edges, key=lambda x:edges[x])[::-1]
for i, gene in enumerate(topk_genes):
    G.add_edge('State 552', gene, weight=round(float(values[i]),4))

# 计算边的权重，设置边的宽度
weights = [G[u][v]['weight'] for u, v in G.edges()]

# 绘制图形
pos = nx.spring_layout(G)  # 选择布局
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='none', edgecolors='none')  # edgecolors='none'去掉边框
nx.draw_networkx_edges(G, pos, width=weights*5, edge_color='steelblue')
nx.draw_networkx_labels(G, pos, font_size=25, font_family='sans-serif')

# 添加边权重标签
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=15)

plt.title("State 552 state-gene interaction", fontsize=30)
plt.axis('off')  # 关闭坐标轴
plt.tight_layout()
plt.savefig('visualize/figures/stategene_552.pdf')