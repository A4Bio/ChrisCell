import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import pandas as pd
import torch
from tqdm import tqdm

gene_names = pd.read_csv('feature.csv')['feature_name'].tolist()
attention, vq_code, gene_indexes = torch.load('tcell_re.pth')
indexes = (vq_code == 552)
attention = attention[indexes][:, 1:, 1:].tolist()
gene_indexes = gene_indexes[indexes]

# 创建一个空的图
G = nx.Graph()
# 添加节点
nodes = []
# 添加边
edges = {}

for k in tqdm(range(47)):
    for i in range(200):
        for j in range(200):
            if i == j:
                continue
            weight = attention[k][i][j]
            # if weight<0.1:
            #     continue
            gene_i = gene_names[gene_indexes[k, i]]
            gene_j = gene_names[gene_indexes[k, j]]
            if gene_i in nodes:
                nodes.append(gene_i)
            if gene_j in nodes:
                nodes.append(gene_j)
            if (gene_i, gene_j) in edges:
                edges[(gene_i, gene_j)] += weight
            elif (gene_j, gene_i) in edges:
                edges[(gene_j, gene_i)] += weight
            else:
                edges[(gene_i, gene_j)] = weight
    
G.add_nodes_from(nodes)
sorted_edges = sorted(edges, key=lambda x:edges[x])[::-1]
for edge in sorted_edges[:100]:
    weight = edges[edge]
    G.add_edge(edge[0], edge[1], weight=weight/100)

# 计算边的权重，设置边的宽度
weights = [G[u][v]['weight'] for u, v in G.edges()]

# 绘制图形
pos = nx.spring_layout(G, k=1.5)  # 选择布局
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='none', edgecolors='none')  # edgecolors='none'去掉边框
nx.draw_networkx_edges(G, pos, width=weights, edge_color='steelblue')
nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif')

# 添加边权重标签
edge_labels = nx.get_edge_attributes(G, 'weight')
filtered_edge_labels = {k: round(edges[k]/100, 3) for k in sorted_edges[:10]}
nx.draw_networkx_edge_labels(G, pos, edge_labels=filtered_edge_labels, font_size=10)

plt.title("State 552 genes interaction", fontsize=30)
plt.axis('off')  # 关闭坐标轴
plt.tight_layout()
plt.savefig('visualize/figures/grnre_552.pdf')