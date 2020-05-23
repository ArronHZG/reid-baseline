import torch
from torch import nn
from torch.autograd.functional import jacobian

from modeling.backbone.autoencoder import AutoEncoder


class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()
        self.bce = nn.MSELoss()

    def forward(self, recon, target):
        return self.bce(recon.to(torch.float), target.to(torch.float).detach())


class AELossL1(nn.Module):
    def __init__(self, lambda_=0.01):
        super(AELossL1, self).__init__()
        self.lambda_ = lambda_
        self.bce = nn.MSELoss()

    def forward(self, model: AutoEncoder, recon, target):
        ae = self.bce(recon.to(torch.float), target.to(torch.float).detach())
        weight = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                weight += torch.abs(m.weight).sum()
        return ae + self.lambda_ * weight


class AELossL2(nn.Module):
    def __init__(self, lambda_=0.01):
        super(AELossL2, self).__init__()
        self.lambda_ = lambda_
        self.bce = nn.MSELoss()

    def forward(self, model: AutoEncoder, recon, target):
        ae = self.bce(recon.to(torch.float), target.to(torch.float).detach())
        weight = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                weight += 0.5 * (m.weight ** 2).sum()
        return ae + self.lambda_ * weight


class CAELoss(nn.Module):
    def __init__(self, lambda_=0.01):
        super(CAELoss, self).__init__()
        self.lambda_ = lambda_
        self.bce = nn.MSELoss()

    def forward(self, model: AutoEncoder, recon, target):
        ae = self.bce(recon.to(torch.float), target.to(torch.float).detach())
        ja = jacobian(model.encoder, target)
        ja = (ja ** 2).sum()
        return ae + self.lambda_ * ja


if __name__ == '__main__':
    m = AutoEncoder(10, 5)
    a = torch.randn(10, 10)
    r = m(a)
    l0 = AELoss()
    l1 = AELossL1()
    l2 = AELossL2()
    lcae = CAELoss()
    l = l0(r, a)
    print(l)
    # l.backward()
    l = l1(m, r, a)
    print(l)
    # l.backward()
    l = l2(m, r, a)
    print(l)
    # l.backward()
    l = lcae(m, r, a)
    print(l)
    # l.backward()
