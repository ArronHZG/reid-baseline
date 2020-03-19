import torch

from modeling.backbones.resnet import ResNet, Bottleneck

resnet50 = ResNet(last_stride=1,
                  block=Bottleneck,
                  layers=[3, 4, 6, 3])

input = torch.randn(2, 3, 256, 128)
print(input.size())
output = resnet50(input)
print(output.size())
