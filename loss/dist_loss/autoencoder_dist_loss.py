import torch
from torch import nn
from torch.nn import functional as F, BCELoss

from modeling.backbone.autoencoder import AutoEncoder


class AutoEncoderDistLoss(nn.Module):
    """
        Distilling the Knowledge in a Neural Network
    """

    def __init__(self, T=10):
        super(AutoEncoderDistLoss, self).__init__()
        self.T = T
        self.mse_loss = BCELoss()

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
        source_item = F.softmax(current, dim=1)

        loss = self.mse_loss(source_item, target_item)
        return loss
