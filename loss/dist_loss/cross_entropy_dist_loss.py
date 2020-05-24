import torch
from torch import nn
from torch.nn import functional as F, BCELoss


class CrossEntropyDistLoss(nn.Module):
    """
        Distilling the Knowledge in a Neural Network
    """

    def __init__(self, T=0.05):
        super(CrossEntropyDistLoss, self).__init__()
        self.T = T
        self.mse_loss = BCELoss()

    def forward(self, current, target):
        target = target / self.T
        target_item = F.softmax(target, dim=1)

        current = current / self.T
        source_item = F.softmax(current, dim=1)

        loss = self.mse_loss(source_item, target_item)
        return loss


if __name__ == '__main__':
    a = torch.randn(10, 20)
    b = torch.randn(10, 20)
    l = CrossEntropyDistLoss(T=10)
    lo = l(a, b)
    print(lo)
