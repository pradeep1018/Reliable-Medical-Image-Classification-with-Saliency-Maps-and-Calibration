import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc

class AUC(nn.Module):
    def __init__(self):
        super(AUC, self).__init__()

    def forward(self, pred, target):
        p_n = pred.cpu().detach().numpy()
        t_n = target.cpu().detach().numpy()
        n_classes = pred.shape[-1]
        auclist = []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(t_n[:, i], p_n[:, i])
            auclist.append(auc(fpr, tpr))
        auc_score = torch.from_numpy(np.array(auclist))
        return auc_score