from copy import copy

import torch
from torch import nn
from torch.autograd.functional import jacobian

from modeling.backbone.autoencoder import AutoEncoder


class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, recon, target):
        return self.mse(recon.to(torch.float), target.to(torch.float).detach())


class AELossL1(nn.Module):
    def __init__(self, lambda_=0.01):
        super(AELossL1, self).__init__()
        self.lambda_ = lambda_
        self.mse = nn.MSELoss()

    def forward(self, model: AutoEncoder, recon, target):
        ae = self.mse(recon, target.detach())
        weight = 0
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                weight += torch.abs(m.weight).sum()
        return ae + self.lambda_ * weight


class AELossL2(nn.Module):
    def __init__(self, lambda_=0.01):
        super(AELossL2, self).__init__()
        self.lambda_ = lambda_
        self.mse = nn.MSELoss()

    def forward(self, model: AutoEncoder, recon, target):
        ae = self.mse(recon, target.detach())
        weight = 0
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                weight += 0.5 * (m.weight ** 2).sum()
        return ae + self.lambda_ * weight


# class CAELoss(nn.Module):
#     def __init__(self, lambda_=0.01):
#         super(CAELoss, self).__init__()
#         self.lambda_ = lambda_
#         self.mse = nn.MSELoss()
#
#     def forward(self, model: AutoEncoder, recon, target):
#         ae = self.mse(recon, target.detach())
#         ja = jacobian(model.encoder, target.detach())
#         ja = (ja ** 2).sum()
#         return ae + self.lambda_ * ja


if __name__ == '__main__':
    m = AutoEncoder(1024, 512).cuda()
    a = torch.randn(4, 1024).cuda()
    r = m(a)
    l0 = AELoss()
    l1 = AELossL1()
    l2 = AELossL2()
    l = l0(r, a)
    print(l)
    l = l1(m, r, a)
    print(l)
    l = l2(m, r, a)
    print(l)
