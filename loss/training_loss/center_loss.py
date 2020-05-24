from __future__ import absolute_import

import torch
from torch import nn

from utils.tensor_utils import euclidean_dist


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim, loss_weight, learning_weight=False):
        super(CenterLoss, self).__init__()
        self.learning_weight = learning_weight
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim), requires_grad=True)
        self.optimizer = None
        if self.learning_weight:
            self.uncertainty = nn.Parameter(torch.tensor(1.64), requires_grad=True)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        dist_mat = euclidean_dist(x, self.centers)

        classes = torch.arange(self.num_classes).long()
        if self.centers.is_cuda:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes)).float()
        dist = dist_mat * mask
        loss = dist.sum() / batch_size

        if self.learning_weight:
            loss = 0.5 * torch.exp(-self.uncertainty) * loss + self.uncertainty
            loss = loss.squeeze(-1)

        return loss


if __name__ == '__main__':
    use_gpu = False
    center_loss = CenterLoss()
    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4])

    loss_ = center_loss(features, targets)
    print(loss_)
