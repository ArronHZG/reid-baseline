import torch
from torch import nn
from torch.nn import functional as F


class CrossEntropyDistLoss(nn.Module):
    """
        distillation loss using cross-entropy
        return -target*log(current)
    """

    def __init__(self, T=0.1):
        super(CrossEntropyDistLoss, self).__init__()
        self.T = T

    def forward(self, current, target):
        target = target / self.T
        target_item = F.softmax(target, dim=1)

        current = current / self.T
        source_item = F.log_softmax(current, dim=1)

        loss = torch.sum(-target_item * source_item, 1).mean()
        return loss
