import torch
from torch import nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_prob = self.log_softmax(inputs)
        one_hot = torch.zeros_like(log_prob).scatter_(1, targets.unsqueeze(1), 1)
        soft_hot = (1 - self.epsilon) * one_hot + self.epsilon / self.num_classes
        cross_entropy = (- soft_hot * log_prob).mean(0).sum()
        return cross_entropy


if __name__ == '__main__':
    i = torch.randn(2, 4).cuda()
    o = torch.randint(4, (2,)).cuda()
    loss = CrossEntropyLabelSmooth(4)
    l = loss(i, o)
    print(l)
