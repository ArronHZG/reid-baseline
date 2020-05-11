import torch
from torch import nn


class MyCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, label_smooth=False, learning_weight=False):
        super(MyCrossEntropy, self).__init__()
        self.label_smooth = label_smooth
        self.learning_weight = learning_weight
        self.num_classes = num_classes
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.epsilon = 0

        if self.learning_weight:
            self.uncertainty = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.optimizer = None
            self.scheduler = None

        if self.label_smooth:
            self.epsilon = 0.1

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

        if self.learning_weight:
            cross_entropy = torch.exp(-self.uncertainty) * cross_entropy + 0.5 * self.uncertainty
            cross_entropy = cross_entropy.squeeze(-1)
        return cross_entropy


if __name__ == '__main__':
    i = torch.randn(2, 4).cuda()
    o = torch.randint(4, (2,)).cuda()
    loss = MyCrossEntropy(4, False, True).cuda()
    l = loss(i, o)
    print(l)

    loss2 = nn.CrossEntropyLoss()
    l2 = loss2(i, o)
    print(l2)
