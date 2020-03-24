import torch


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
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
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
