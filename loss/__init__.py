# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import logging

from torch.nn import CrossEntropyLoss

from .dec_loss import DECLoss
from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss


class Loss:
    def __init__(self, cfg, num_classes, feat_dim=2048, device='cpu'):  # modified by arron

        self.cfg = cfg
        self.num_classes = num_classes
        self.loss_type = cfg.LOSS.LOSS_TYPE
        using_gpu = cfg.MODEL.DEVICE == 'cuda'
        if self.cfg.LOSS.IF_LABEL_SMOOTH:
            self.xent = CrossEntropyLabelSmooth(num_classes=self.num_classes,
                                                use_gpu=using_gpu)
            logger = logging.getLogger("reid_baseline")
            logger.info(f"Label smooth on, numclasses: {self.num_classes}")
        else:
            self.xent = CrossEntropyLoss()

        self.triplet = TripletLoss(self.cfg.LOSS.MARGIN)

        self.using_center = cfg.LOSS.IF_WITH_CENTER
        self.using_dec = cfg.LOSS.DEC

        if self.using_center:
            self.center_loss_weight = cfg.LOSS.CENTER_LOSS_WEIGHT
            self.center = CenterLoss(num_classes=self.num_classes, feat_dim=feat_dim)
            self.center.to(device)
            if self.using_dec:
                self.dec = DECLoss()
        else:
            self.center = None

    def make_loss(self):
        if self.using_center:
            return self._loss_with_center()
        else:
            return self._loss()

    def _loss(self):

        loss_function = None

        if self.loss_type == 'softmax':
            def loss_function(score, feat, target):
                return self.xent(score, target)

        elif self.loss_type == 'triplet':
            def loss_function(score, feat, target):
                return self.triplet(feat, target)

        elif self.loss_type == 'softmax_triplet':

            def loss_function(score, feat, target):
                return self.xent(score, target) + self.triplet(feat, target)
        else:
            print('expected sampler should be softmax, triplet or softmax_triplet, '
                  'but got {}'.format(self.loss_type))

        return loss_function

    def _loss_with_center(self):

        loss_function = None

        if self.loss_type == 'softmax':
            def loss_function(score, feat, target):
                return self.xent(score, target) + self.center_loss_weight * self.center(feat, target)

        elif self.loss_type == 'triplet':
            def loss_function(score, feat, target):
                return self.triplet(feat, target) + self.center_loss_weight * self.center(feat, target)

        elif self.loss_type == 'softmax_triplet':
            def loss_function(score, feat, target):
                return self.xent(score, target) + self.triplet(feat, target) + self.center_loss_weight * self.center(
                    feat, target)
        else:
            print('expected sampler should be softmax, triplet or softmax_triplet, '
                  'but got {}'.format(self.loss_type))

        return loss_function

    def _loss_with_center_and_dec(self):

        loss_function = None

        if self.loss_type == 'softmax':
            def loss_function(score, feat, target):
                return self.xent(score, target) + self.center_loss_weight * self.center(feat, target)

        elif self.loss_type == 'triplet':
            def loss_function(score, feat, target):
                return self.triplet(feat, target) + self.center_loss_weight * self.center(feat, target)

        elif self.loss_type == 'softmax_triplet':
            def loss_function(score, feat, target):
                return self.xent(score, target) + self.triplet(feat, target) + self.center_loss_weight * self.center(
                    feat, target)
        else:
            print('expected sampler should be softmax, triplet or softmax_triplet, '
                  'but got {}'.format(self.loss_type))

        return loss_function
