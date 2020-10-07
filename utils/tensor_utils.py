import torch
from torch.cuda.amp import autocast


def batch_horizontal_flip(tensor):
    """
    :param tensor: N x C x H x W
    :return:
    """
    inv_idx = torch.arange(tensor.size(3) - 1, -1, -1).long().cuda()
    img_flip = tensor.index_select(3, inv_idx)
    return img_flip


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(beta=1, alpha=-2, mat1=x, mat2=y.t())
    dist.clamp_(min=1e-12)
    dist.sqrt_()  # for numerical stability
    return dist


def cosine_dist(x, y):
    dist = 1 - torch.mm(x, y.t())
    dist.clamp_(min=1e-12)
    return dist


if __name__ == '__main__':
    a = torch.tensor([[0., 0.]])
    b = torch.tensor([[1., 1.]])
    dist = euclidean_dist(a, b)
    print(dist)

    a = torch.randn(4, 2048, 16, 4)
    b = torch.tensor([[1., 1.]])
    dist = euclidean_dist(a, b)
    print(dist)
