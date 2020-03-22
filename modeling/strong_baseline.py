from torch import nn

from modeling.backbone.resnet import resnet18, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, \
    wide_resnet50_2, wide_resnet101_2
from modeling.model_initial import weights_init_kaiming, weights_init_classifier

model_map = {'resnet18': resnet18,
             'resnet50': resnet50,
             'resnet101': resnet101,
             'resnet152': resnet152,
             'resnext50_32x4d': resnext50_32x4d,
             'resnext101_32x8d': resnext101_32x8d,
             'wide_resnet50_2': wide_resnet50_2,
             'wide_resnet101_2': wide_resnet101_2}


# TODO se_resnet
# TODO ibn_resnet

class Baseline(nn.Module):

    def __init__(self, num_classes, last_stride, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        self.base = model_map[model_name](last_stride=last_stride,
                                          pretrained=True if pretrain_choice == 'imagenet' else False)

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.in_planes = 512 * self.base.block.expansion

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
        feat_t = self.GAP(x).view(x.size(0), -1)
        feat_c = self.bottleneck(feat_t)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat_c)
            return feat_t, cls_score  # global feature for triplet loss
        else:  # Test with feature after BN
            return feat_c
