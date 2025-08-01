import os
import sys
import warnings

# Ignore all FutureWarning warnings
warnings.filterwarnings("ignore", category=Warning)
from sklearn.cluster import SpectralClustering

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from module.dataloader import *
from tqdm import tqdm as tq
import time

import torch
import numpy as np
from utils.util_print import *
from model import HomoGraph
from module.AutoLR import WarmUpLR, downLR
from utils.util_loss import *
from utils.utils_gac import load_data
import random

# EPOCH, FineEpoch, TrainLR, FineLR = 300, 100, 5e-4, 1e-4
EPOCH, FineEpoch, TrainLR, FineLR = 300, 100, 5e-4, 1e-4


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])



def finetune(model, count, y, centers, Cell_X, Cell_Adj, Gene_X, Gene_Adj, device=None):
    optimizer, criterion = torch.optim.Adam(params=model.parameters(), lr=FineLR), nn.MSELoss()
    warm_up_sche = WarmUpLR(optimizer, total_iters=0.5 * FineEpoch)
    down_sche = downLR(optimizer, total_iters=0.5 * FineEpoch)
    p = 100
    model.cluster_model.clusters = torch.nn.Parameter(centers, requires_grad=True)
    for epoch in tq(range(FineEpoch)):
        Cell, Gene, Mat, q, p, pi, disp, mean = model.fintune(Cell_X, Gene_X, epoch, p, n_update=8)
        inputs = [Cell, Gene, Cell_X, Cell_Adj, Gene_Adj, count, Mat, p, q, criterion, pi, disp, mean]
        FinetuneLoss, FinetuneWeight = Finetune_loss(*inputs)
        optimizer.zero_grad()
        loss = FinetuneLoss[-1]
        loss.backward()
        optimizer.step()
        if epoch < 0.5 * FineEpoch:
            warm_up_sche.step()
        else:
            down_sche.step()
        if epoch % 10 == 0:
            print_metrics(y, np.argmax(q.detach().cpu().numpy(), axis=1), epoch, FinetuneLoss)
            
        if torch.isnan(loss):
            break
    nmi, ari = print_metrics(y, np.argmax(q.detach().cpu().numpy(), axis=1), epoch, FinetuneLoss)
    return model, nmi, ari


def train(count, y, A=None, X=None, device=None):
    cluster_number = int(max(y) - min(y) + 1)
    if A is None:
        Cell_X, Cell_Adj = Data(count)[0], Data(count,False)[1]
        Cell_X = Cell_X.to(device)
        model = HomoGraph.HomoGraph(count, Cell_X, cluster_number).to(device)
        Gene_X, Gene_Adj = Data(count.T)[0], Data(count.T, False)[1]
        Gene_X = Gene_X.to(device)
    else:
        Cell_X = HomoData(X, A).to(device)
        Cell_Adj = A.copy()
        model = HomoGraph.HomoGraph(X, A, cluster_number).to(device)
        Gene_X, Gene_Adj = Data(X.T)[0], Data(X.T, False)[1]
        Gene_X = Gene_X.to(device)
        count = X.copy()
    model.train()
    optimizer, criterion = torch.optim.Adam(params=model.parameters(), lr=TrainLR), nn.MSELoss()
    Cell_Adj = torch.tensor(Cell_Adj, dtype=torch.float).to(device)
    Gene_Adj = torch.tensor(Gene_Adj, dtype=torch.float).to(device)
    for epoch in tq(range(EPOCH)):
        Cell, Gene, rec_X, zmean, pi, disp, mean = model(Cell_X, Gene_X)
        TrainLoss = Train_loss(Cell, Gene, Cell_X, Cell_Adj, Gene_Adj, count, rec_X, criterion, pi, disp, mean)
        loss = TrainLoss[-1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if torch.isnan(loss):
            break
    if not torch.isnan(loss):
        labels = SpectralClustering(n_clusters=cluster_number, affinity="precomputed", assign_labels="discretize",
                                    random_state=0).fit_predict(Cell_Adj.cpu())
        centers = computeCentroids(zmean.cpu().detach().numpy(), labels)
        centers = torch.tensor(centers, dtype=torch.float).cuda(device)
        model, nmi, ari = finetune(model, count, y, centers, Cell_X, Cell_Adj, Gene_X, Gene_Adj, device)
        return nmi, ari
    else: 
        return 0, 0


if __name__ == '__main__':
    start = time.time()
    torch.cuda.set_device("cuda:0")
    device = torch.device("cuda:0")
    dataname = "Quake_10x_Limb_Muscle"
    print("Data input is ", dataname)
    seed = 6487
    seed_all(seed)
    filter_count, y, count = read_count(dataname, 500)
    if count.shape[0] < 1000:
        data_path = 'data/' + dataname + '/data.tsv'
        A, X, cells, genes = load_data(data_path, dataname,
                                        512, True, int(max(y) - min(y) + 1), 15)
        print(count.shape, len(y), A.shape, X.shape)
        nmi,ari = train(filter_count, y, A, X, device)  
    else:
        print(count.shape, len(y))
        nmi,ari = train(filter_count, y, None, None, device)  
    end = time.time()
    print("Time cost:", end - start)






