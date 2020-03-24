import torch

loss_list = []


def fun(x1, x2, x3):
    return x1 + x2


def fun2(x2, x3, x4):
    return x2 + x3 * x4


loss_list.append(fun)
loss_list.append(fun2)

a = torch.tensor(.0, requires_grad=True)
x1 = torch.tensor(1., requires_grad=True)
x2 = torch.tensor(2., requires_grad=True)
x3 = torch.tensor(3., requires_grad=True)
for loss in loss_list:
    temp = loss(x1, x2, x3)
    a = a + temp
print(a)
a.backward()