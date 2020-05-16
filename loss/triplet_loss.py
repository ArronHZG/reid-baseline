# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
from torch.nn import TripletMarginLoss

from utils.tensor_utils import euclidean_dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(nn.Module):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=0.3, learning_weight=False):
        super(TripletLoss, self).__init__()
        self.learning_weight = learning_weight
        self.margin = margin
        # max(0,-y(x1-x2)+margin)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

        if self.learning_weight:
            self.uncertainty = nn.Parameter(torch.tensor(2.73), requires_grad=True)
            self.optimizer = None

    def forward(self, global_feat, labels):
        global_feat = torch.nn.functional.normalize(global_feat, dim=1, p=2)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        # y = dist_an.new().resize_as_(dist_an).fill_(1)
        ################################################################################
        # Person re-identification by multi-channel parts-based CNN with improved triplet loss function.
        # loss = ap + (ap - an + mergin)+
        ################################################################################
        # zero = torch.zeros_like(dist_an)
        # ap_an_margin = dist_ap - dist_an + self.margin
        # ap_an_margin = torch.max(torch.stack((ap_an_margin, zero)), 0)
        # loss = (dist_ap + ap_an_margin[0]).mean()

        # If y = 1  then it assumed the first input should be ranked higher
        # (have a larger value) than the second input,
        # and vice-versa for y = -1.
        y = torch.zeros_like(dist_an).fill_(1)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        if self.learning_weight:
            loss = 0.5 * torch.exp(-self.uncertainty) * loss + self.uncertainty
            loss = loss.squeeze(-1)

        # return loss, dist_ap, dist_an
        return loss


if __name__ == '__main__':
    feat = torch.randn(6, 100)
    label = torch.tensor([1, 1, 1, 2, 2, 2]).int()
    tr = TripletLoss(margin=0.3)
    lo = tr(feat, label)
    print(lo)
    TripletMarginLoss()
