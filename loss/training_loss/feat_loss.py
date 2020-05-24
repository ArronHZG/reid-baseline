from torch import nn


class FeatLoss(nn.Module):
    def __init__(self):
        super(FeatLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, input_feat, target_feat):
        return self.l1_loss(input_feat, target_feat)
