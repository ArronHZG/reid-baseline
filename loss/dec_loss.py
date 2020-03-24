import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional


class DECLoss(nn.Module):
    def __init__(self):
        super(DECLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, batch: torch.Tensor, cluster_centers: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - cluster_centers) ** 2, 2)
        numerator = (1.0 + (norm_squared / self.alpha)) ** -1.0
        power = -(self.alpha + 1) / 2
        numerator = numerator ** power
        output = (numerator.t() / torch.sum(numerator, 1)).t()
        target_p = self.target_distribution(output)
        loss = self.kl(output.log(), target_p)
        return loss

    @staticmethod
    def target_distribution(batch: torch.Tensor) -> torch.Tensor:
        """
        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


if __name__ == '__main__':
    clu = ClusterAssignment(10, 2048)
    x = torch.randn(2, 2048)
    initial_cluster_centers = torch.randn(751, 2048)
    nn.init.xavier_uniform_(initial_cluster_centers)

    out = clu(x, initial_cluster_centers)
    print(out)
