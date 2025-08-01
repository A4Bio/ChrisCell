from sklearn import metrics

from module.graph_function import *
from utils.utils import cluster_acc


def cal_cluster_metrics(y, y_pred):
    acc = np.round(cluster_acc(y, y_pred), 5)
    y = list(map(int, y))
    y_pred = np.array(y_pred)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    return acc, nmi, ari


Metric_Name = ['Cell_loss', 'Gene_loss', 'Mat_loss', 'zinbloss', 'sinkhorn_div', 'loss']


def print_metrics(y, y_pred, epoch, Loss, display_loss = False, Weight = None):
    acc, nmi, ari = cal_cluster_metrics(y, y_pred)
    if display_loss:
        Loss = [Loss[i] * Weight[i] for i in range(len(Loss) - 1)]
        Name = [Metric_Name[i] + ': {:.4f}'.format(Loss[i]) for i in range(len(Loss)) if Loss[i] > 0]
        print('Epoch: {:04d}'.format(epoch + 1), ' '.join(Name), 'ACC: {:.4f}'.format(acc),
              'NMI: {:.4f}'.format(nmi), 'ARI: {:.4f}'.format(ari))
    else:
        print('Epoch: {:04d}'.format(epoch + 1), 'ACC: {:.4f}'.format(acc),
              'NMI: {:.4f}'.format(nmi), 'ARI: {:.4f}'.format(ari))
    return nmi, ari

