from torch import nn

from modeling.strong_baseline import Baseline
from modeling.backbone.autoencoder import AutoEncoder
from utils import Data


class EBLL(nn.Module):
    def __init__(self,
                 num_classes,
                 last_stride,
                 model_name,
                 pretrain_choice,
                 se=False,
                 ibn_a=False,
                 ibn_b=False,
                 code_size=1024):
        super(EBLL, self).__init__()
        self.baseline = Baseline(num_classes,
                                 last_stride,
                                 model_name,
                                 pretrain_choice,
                                 se=se,
                                 ibn_a=ibn_a,
                                 ibn_b=ibn_b)
        self.in_planes = self.baseline.in_planes
        self.ae = AutoEncoder(self.in_planes, code_size)
        self.ael1 = AutoEncoder(self.in_planes, code_size)
        self.ael2 = AutoEncoder(self.in_planes, code_size)
        self.cae = AutoEncoder(self.in_planes, code_size)

    def forward(self, x) -> Data:
        data = self.baseline(x)
        data.recon_ae = self.ae(data.feat_t)
        data.ae = self.ae

        data.recon_ael1 = self.ael1(data.feat_t)
        data.ael1 = self.ael1

        data.recon_ael2 = self.ael2(data.feat_t)
        data.ael2 = self.ael2

        data.recon_cae = self.cae(data.feat_t)
        data.cae = self.cae
        return data
