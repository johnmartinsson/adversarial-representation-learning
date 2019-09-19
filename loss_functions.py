import torch
import torch.nn as nn
import torch.nn.functional as F

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        c = torch.tensor(x.shape[1]).float()
        n = torch.tensor(x.shape[0]).float()
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return (b / torch.log(c))/n
