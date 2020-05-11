import logging
from collections import OrderedDict

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from loss.arcface_loss import ArcfaceLoss
from loss.center_loss import CenterLoss
from loss.dec_loss import DECLoss
from loss.smoth_loss import MyCrossEntropy
from loss.triplet_loss import TripletLoss
from loss.dist_loss import CrossEntropyDistLoss

logger = logging.getLogger("reid_baseline.loss")


class Loss:
    def __init__(self, cfg, num_classes, feat_dim):

        self.loss_type = cfg.LOSS.LOSS_TYPE
        self.loss_function_map = OrderedDict()

        # loss_function **kw should have:
        #     feat_t,
        #     feat_c,
        #     cls_score,
        #     target,
        #     target_feat_c,

        # ID loss
        self.xent = None
        if 'softmax' in self.loss_type:
            self.xent = MyCrossEntropy(num_classes=num_classes,
                                       label_smooth=cfg.LOSS.IF_LABEL_SMOOTH,
                                       learning_weight=cfg.LOSS.IF_LEARNING_WEIGHT)

            if cfg.MODEL.DEVICE is 'cuda':
                self.xent = self.xent.cuda()

            if self.xent.learning_weight:
                self.xent.optimizer = torch.optim.SGD(self.xent.parameters(),
                                                      lr=cfg.OPTIMIZER.LOSS_LR,
                                                      momentum=0.9,
                                                      weight_decay=10 ** -4,
                                                      nesterov=True)
                self.xent.scheduler = ExponentialLR(self.xent.optimizer,
                                                    gamma=0.95,
                                                    last_epoch=-1)

            def loss_function(**kw):
                return self.xent(kw['cls_score'], kw['target'])

            self.loss_function_map["softmax"] = loss_function

        if 'arcface' in self.loss_type:
            self.arcface = ArcfaceLoss(num_classes=num_classes, feat_dim=feat_dim)

            if cfg.MODEL.DEVICE is 'cuda':
                self.arcface = self.arcface.cuda()

            def loss_function(**kw):
                return self.arcface(kw['feat_c'], kw['target'])

            self.loss_function_map["arcface"] = loss_function

        # metric loss
        self.triplet = None
        if 'triplet' in self.loss_type:
            self.triplet = TripletLoss(cfg.LOSS.MARGIN,
                                       learning_weight=False)

            if cfg.MODEL.DEVICE is 'cuda':
                self.triplet = self.triplet.cuda()

            if self.triplet.learning_weight:
                self.triplet.optimizer = torch.optim.SGD(self.triplet.parameters(), lr=cfg.OPTIMIZER.LOSS_LR)

            def loss_function(**kw):
                return cfg.LOSS.METRIC_LOSS_WEIGHT * self.triplet(kw['feat_t'], kw['target'])

            self.loss_function_map["triplet"] = loss_function

        # cluster loss
        self.center = None
        if cfg.LOSS.IF_WITH_CENTER:
            self.center = CenterLoss(num_classes=num_classes,
                                     feat_dim=feat_dim,
                                     loss_weight=cfg.LOSS.CENTER_LOSS_WEIGHT,
                                     learning_weight=False)

            if cfg.MODEL.DEVICE is 'cuda':
                self.center = self.center.cuda()
            self.center.optimizer = torch.optim.SGD(self.center.parameters(),
                                                    lr=cfg.OPTIMIZER.LOSS_LR,
                                                    momentum=0.9,
                                                    weight_decay=10 ** -4,
                                                    nesterov=True)

            self.center.scheduler = ExponentialLR(self.center.optimizer,
                                                  gamma=0.995,
                                                  last_epoch=-1)

            def loss_function(**kw):
                return cfg.LOSS.CENTER_LOSS_WEIGHT * self.center(kw['feat_t'], kw['target'])

            self.loss_function_map["center"] = loss_function

            if cfg.LOSS.IF_WITH_DEC:
                self.dec = DECLoss()

                def loss_function(**kw):
                    return self.dec(kw['feat_t'], self.center.centers)

                self.loss_function_map["dec"] = loss_function

        # dist loss
        self.cross_entropy_dist_loss = None
        if cfg.CONTINUATION.IF_ON:
            self.cross_entropy_dist_loss = CrossEntropyDistLoss(T=cfg.CONTINUATION.T)

            def loss_function(**kw):
                return self.cross_entropy_dist_loss(kw['feat_c'], kw['target_feat_c'])

            self.loss_function_map["dist"] = loss_function

    def optimizer_zero_grad(self):
        if self.center:
            self.center.optimizer.zero_grad()

        if self.xent and self.xent.learning_weight:
            self.xent.optimizer.zero_grad()

        if self.triplet and self.triplet.learning_weight:
            self.triplet.optimizer.zero_grad()

    def optimizer_step(self):
        if self.center:
            for param in self.center.parameters():
                param.grad.data *= (1. / self.center.loss_weight)
            self.center.optimizer.step()

        if self.xent and self.xent.learning_weight:
            self.xent.optimizer.step()

        if self.triplet and self.triplet.learning_weight:
            self.triplet.optimizer.step()

    def scheduler_step(self):
        if self.center:
            self.center.scheduler.step()

        if self.xent and self.xent.learning_weight:
            self.xent.scheduler.step()
