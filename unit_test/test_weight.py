import torch
weight = torch.load('/home/hzg/PycharmProjects/reid-baseline/run/ebll/dukemtmc/resnet50/experiment-05/train_checkpoint_36150.pth')

print(weight.keys())
