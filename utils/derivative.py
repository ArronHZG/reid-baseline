import torch
from torch.autograd import grad
from torch.autograd.functional import jacobian


def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()
    return grads


x = torch.arange(4.0, requires_grad=True).reshape(2, 2)
print(x)


def exp_reducer(x):
    return x.exp().mean()


print(nth_derivative(f=exp_reducer(x), wrt=x, n=1))

j = jacobian(exp_reducer, x)
print(j)
