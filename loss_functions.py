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

def mean_weighted_squared_error(x1, x2, w, C, lambd):
    # penalize area larger than C
    penalty = hparams.lambd*torch.pow(F.relu(torch.mean(w)-C), 2)
    
    w = torch.full(w.shape, 1.0).to(hparams.device) - w # broadcast
    return torch.mean(torch.pow(x1-x2, 2) * w) + penalty

def mean_weighted_absolute_error(x1, x2, w, C, lambd):
    # penalize area larger than C
    penalty = hparams.lambd*torch.pow(F.relu(torch.mean(w)-C), 2)

    writer.add_scalar('loss/weights', torch.mean(w).cpu().detach().numpy(), training_batch_counter)
    
    w = torch.full(w.shape, 1.0).to(hparams.device) - w # broadcast
    return torch.mean(torch.abs(x1-x2) * w) + penalty

def mean_absolute_error(x1, x2):
    return torch.mean(torch.abs(x1-x2))

def mean_squared_error(x1, x2):
    return torch.mean(torch.pow(x1-x2, 2))


