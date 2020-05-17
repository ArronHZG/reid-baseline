import torch
from torch import nn

from loss.triplet_loss import hard_example_mining
from utils.tensor_utils import euclidean_dist


class TripletDistLoss(nn.Module):
    """
        distillation loss using triplet loss
        source_ap, source_an
        current_ap, current_an
        (max(sap,cap)-min(san,can)+alpha)+
    """

    def __init__(self, margin=0.3, T=0.05):
        super(TripletDistLoss, self).__init__()
        self.T = T
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, current, target, labels):
        current_global_feat = torch.nn.functional.normalize(current, dim=1, p=2)
        current_dist_mat = euclidean_dist(current_global_feat, current_global_feat)
        current_dist_ap, current_dist_an = hard_example_mining(current_dist_mat, labels)

        target_global_feat = torch.nn.functional.normalize(target, dim=1, p=2)
        target_dist_mat = euclidean_dist(target_global_feat, target_global_feat)
        target_dist_ap, target_dist_an = hard_example_mining(target_dist_mat, labels)

        max_ap, _ = torch.max(torch.stack((current_dist_ap, target_dist_ap)), 0)
        min_an, _ = torch.min(torch.stack((current_dist_an, target_dist_an)), 0)

        y = torch.zeros_like(max_ap).fill_(1)
        loss = self.ranking_loss(min_an, max_ap, y)
        return loss


if __name__ == '__main__':
    current = torch.randn(6, 100)
    target = torch.randn(6, 100)
    label = torch.tensor([1, 1, 1, 2, 2, 2]).int()
    tr = TripletDistLoss(margin=0.3)
    lo = tr(current, target, label)
    print(lo)
