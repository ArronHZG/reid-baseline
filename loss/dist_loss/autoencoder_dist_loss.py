import torch
from torch import nn
from torch.nn import functional as F, BCELoss, MSELoss

from modeling.backbone.autoencoder import AutoEncoder


class BCEAutoEncoderDistLoss(nn.Module):
    """
        Distilling the Knowledge in a Neural Network
    """

    def __init__(self, T=10):
        super(BCEAutoEncoderDistLoss, self).__init__()
        self.T = T
        self.bce_loss = BCELoss()

    def forward(self,
                current_model: AutoEncoder,
                source_model: AutoEncoder,
                current,
                target):
        current = current_model.encoder(current)
        target = source_model.encoder(target).detach()

        target = target / self.T
        target_item = F.softmax(target, dim=1)

        current = current / self.T
        current_item = F.softmax(current, dim=1)

        loss = self.bce_loss(current_item, target_item)
        return loss


class MSEAutoEncoderDistLoss(nn.Module):

    def __init__(self):
        super(MSEAutoEncoderDistLoss, self).__init__()
        self.mse_loss = MSELoss()

    def forward(self,
                current_model: AutoEncoder,
                source_model: AutoEncoder,
                current,
                target):
        current = current_model.encoder(current)
        target = source_model.encoder(target).detach()

        loss = self.mse_loss(current, target)
        return loss