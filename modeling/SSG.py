import torch

from loss.dec_loss import ClusterAssignment
from modeling.strong_baseline import Baseline


class SSG(Baseline):
    def __init__(self, num_classes, last_stride=1, if_bnneck=True,
                 neck_feat='after', model_name='resnet50',
                 pretrain_choice='ibn-net'):
        super(SSG, self).__init__(num_classes, last_stride,
                                  if_bnneck, neck_feat,
                                  model_name, pretrain_choice)

        self.assignment = ClusterAssignment(cluster_number=32, embedding_dimension=self.in_planes)

    def forward(self, x):
        whole = self.base(x)
        half_high = int(whole.size(2) / 2)

        up = whole[:, :, half_high:, :]
        down = whole[:, :, :half_high, :]

        whole = self.GAP(whole).view(whole.size(0), -1)
        up = self.GAP(up).view(up.size(0), -1)
        down = self.GAP(down).view(down.size(0), -1)

        print(whole.size())
        print(up.size())
        print(down.size())

        feat_t = [whole, up, down]
        feat_c = self.bottleneck(feat_t)
        if self.training:
            x = self.assignment(torch.cat(feat_t, dim=1))
            return feat_t, feat_c, x
        else:
            feat_t = torch.cat(feat_t, dim=1)
            return feat_t, feat_c


if __name__ == '__main__':
    image = torch.randn(2, 3, 256, 128)

    ssg = SSG(100)
    output = ssg(image)
