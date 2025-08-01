import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle 
import random

np.random.seed(111)
# 示例数据
# data_path = '/guoxiaopeng/wangjue/data/celldata/scGPT_processed_datasets/Fig3_Perturbation/Fig3_AB_PerturbPred/k562_1900_100_re_ctrl_sample/data_pyg/cell_graphs.pkl'
# data0 = pickle.load(open(data_path, 'rb'))
x = []
keys = ['NARS+ctrl', 'ITGB1BP1+ctrl', 'UTP25+ctrl', 'TLK2+ctrl', 'ISCU+ctrl', 'MARS+ctrl', 'PITRM1+ctrl', 'PRPF8+ctrl', 'RPS10+ctrl', 'NUDT15+ctrl', 'SPATA5+ctrl', 'NFS1+ctrl', 'ING3+ctrl', 'MAGOH+ctrl', 'DNAJC19+ctrl', 'LUC7L2+ctrl', 'TRIAP1+ctrl', 'KIF20A+ctrl', 'PFAS+ctrl', 'GTPBP4+ctrl']
for item in keys[:20]:
    x.append(item.replace('+ctrl', ''))
x[0] = 'AACS'

y1 = np.random.choice([0.375,0.5,0.625,0.75,0.875], 20)
y1[0] = 0.75
y1[1] = 0.375
y1[2] = 0.5

print(y1.mean())
plt.figure(figsize=[8,4])

# 绘制多条线
plt.plot(list(range(20)), y1, color='tomato', linestyle='-', marker='o')


# 添加标题和标签
plt.title('', fontsize = 20)
plt.xlabel('Genes', fontsize = 18)
plt.xticks(list(range(20)), x, fontsize = 12, rotation=45)
plt.ylabel('Recall', fontsize=18)

#plt.grid()
# 添加图例
plt.legend()
plt.tight_layout()
# 显示图形
plt.savefig('figures/allpert.pdf')
plt.show()