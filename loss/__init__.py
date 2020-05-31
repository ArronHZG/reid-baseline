import logging
from collections import OrderedDict

import torch
from torch.optim.lr_scheduler import ExponentialLR

from loss.dist_loss.autoencoder_dist_loss import MSEAutoEncoderDistLoss, BCEAutoEncoderDistLoss
from loss.dist_loss.cross_entropy_dist_loss import CrossEntropyDistLoss
from loss.dist_loss.triplet_dist_loss import TripletDistLoss
from loss.training_loss.arcface_loss import ArcfaceLoss
from loss.training_loss.autoencoder_loss import AELoss, AELossL1, AELossL2
from loss.training_loss.center_loss import CenterLoss
from loss.training_loss.dec_loss import DECLoss
from loss.training_loss.smoth_loss import MyCrossEntropy
from loss.training_loss.triplet_loss import TripletLoss
from utils import Data

logger = logging.getLogger("reid_baseline.loss")


class Loss:
    def __init__(self, cfg, num_classes, feat_dim):

        self.loss_type = cfg.LOSS.LOSS_TYPE
        self.loss_function_map = OrderedDict()

        # loss_function should input Data

        # ID loss
        self.xent = None
        if 'softmax' in self.loss_type:
            self.loss_function_map["softmax"] = self.get_softmax_loss(cfg, num_classes)

        if 'arcface' in self.loss_type:
            self.loss_function_map["arcface"] = self.get_arcface_loss(cfg, feat_dim, num_classes)

        # metric loss
        self.triplet = None
        if 'triplet' in self.loss_type:
            self.loss_function_map["triplet"] = self.get_triplet_loss(cfg)

        # cluster loss
        self.center = None
        if cfg.LOSS.IF_WITH_CENTER:
            self.loss_function_map["center"] = self.get_center_loss(cfg, feat_dim, num_classes)
            if cfg.LOSS.IF_WITH_DEC:
                self.loss_function_map["dec"] = self.get_dec_loss()

        # dist loss
        if cfg.CONTINUATION.IF_ON and "ce_dist" in cfg.CONTINUATION.LOSS_TYPE:
            self.loss_function_map["ce_dist"] = self.get_cross_entropy_dist_loss(cfg)

        if cfg.CONTINUATION.IF_ON and "tr_dist" in cfg.CONTINUATION.LOSS_TYPE:
            self.loss_function_map["tr_dist"] = self.get_triplet_dist_loss(cfg)

        if cfg.CONTINUATION.IF_ON and cfg.EBLL.IF_ON:
            self.loss_function_map["ae_dist"] = self.get_code_dist_loss(cfg)

    def __str__(self):
        return "Loss Type: " + " ".join(self.loss_function_map.keys())

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

    def get_triplet_dist_loss(self, cfg):
        self.triplet_dist_loss = TripletDistLoss(T=cfg.CONTINUATION.T)

        def loss_function(data: Data):
            return self.triplet_dist_loss(data.feat_c, data.source.feat_c, data.cls_label)

        return loss_function

    def get_cross_entropy_dist_loss(self, cfg):
        self.cross_entropy_dist_loss = CrossEntropyDistLoss(T=cfg.CONTINUATION.T)

        def loss_function(data: Data):
            return self.cross_entropy_dist_loss(data.feat_c, data.source.feat_c)

        return loss_function

    def get_dec_loss(self):
        self.dec = DECLoss()

        def loss_function(data: Data):
            return self.dec(data.feat_t, self.center.centers)

        return loss_function

    def get_center_loss(self, cfg, feat_dim, num_classes):
        self.center = CenterLoss(num_classes=num_classes,
                                 feat_dim=feat_dim,
                                 loss_weight=cfg.LOSS.CENTER_LOSS_WEIGHT,
                                 learning_weight=False)
        self.center.optimizer = torch.optim.SGD(self.center.parameters(),
                                                lr=cfg.OPTIMIZER.LOSS_LR,
                                                momentum=0.9,
                                                weight_decay=10 ** -4,
                                                nesterov=True)
        self.center.scheduler = ExponentialLR(self.center.optimizer,
                                              gamma=0.995,
                                              last_epoch=-1)
        if cfg.MODEL.DEVICE is 'cuda':
            self.center = self.center.cuda()
            if cfg.APEX.IF_ON:
                self.center.to(torch.half)

        def loss_function(data: Data):
            return cfg.LOSS.CENTER_LOSS_WEIGHT * self.center(data.feat_t, data.cls_label)

        return loss_function

    def get_triplet_loss(self, cfg):
        self.triplet = TripletLoss(cfg.LOSS.MARGIN,
                                   learning_weight=False)
        if cfg.MODEL.DEVICE is 'cuda':
            self.triplet = self.triplet.cuda()
        if self.triplet.learning_weight:
            self.triplet.optimizer = torch.optim.SGD(self.triplet.parameters(),
                                                     lr=0.0001,
                                                     momentum=0.9,
                                                     weight_decay=10 ** -4,
                                                     nesterov=True)
            self.triplet.scheduler = ExponentialLR(self.triplet.optimizer,
                                                   gamma=0.95,
                                                   last_epoch=-1)

        def loss_function(data: Data):
            return cfg.LOSS.METRIC_LOSS_WEIGHT * self.triplet(data.feat_t, data.cls_label)

        return loss_function

    def get_arcface_loss(self, cfg, feat_dim, num_classes):
        self.arcface = ArcfaceLoss(num_classes=num_classes, feat_dim=feat_dim)
        if cfg.MODEL.DEVICE is 'cuda':
            self.arcface = self.arcface.cuda()

        def loss_function(data: Data):
            return self.arcface(data.feat_c, data.cls_label)

        return loss_function

    def get_softmax_loss(self, cfg, num_classes):
        self.xent = MyCrossEntropy(num_classes=num_classes,
                                   label_smooth=cfg.LOSS.IF_LABEL_SMOOTH,
                                   learning_weight=cfg.LOSS.IF_LEARNING_WEIGHT)
        if cfg.MODEL.DEVICE is 'cuda':
            self.xent = self.xent.cuda()
        if self.xent.learning_weight:
            self.xent.optimizer = torch.optim.SGD(self.xent.parameters(),
                                                  lr=0.0001,
                                                  momentum=0.9,
                                                  weight_decay=10 ** -4,
                                                  nesterov=True)
            self.xent.scheduler = ExponentialLR(self.xent.optimizer,
                                                gamma=0.95,
                                                last_epoch=-1)

        def loss_function(data: Data):
            return cfg.LOSS.ID_LOSS_WEIGHT * self.xent(data.cls_score, data.cls_label)

        return loss_function

    def get_code_dist_loss(self, cfg):
        if cfg.EBLL.DIST_TYPE == "bce":
            ae_dist = BCEAutoEncoderDistLoss()
        elif cfg.EBLL.DIST_TYPE == "mse":
            ae_dist = MSEAutoEncoderDistLoss()

        def loss_function(data: Data):
            return ae_dist(data.source.ae, data.feat_t, data.source.feat_t)

        return loss_function


class AutoEncoderLoss:
    def __init__(self, cfg):

        self.loss_type = cfg.LOSS.LOSS_TYPE
        self.loss_function_map = OrderedDict()
        self.xent = None

        # autoencoder loss
        if cfg.EBLL.IF_ON and "ae_loss" in cfg.EBLL.LOSS_TYPE:
            self.loss_function_map["ae"] = self.get_ae_loss()

        if cfg.EBLL.IF_ON and "ae_l1" in cfg.EBLL.LOSS_TYPE:
            self.loss_function_map["ae_l1"] = self.get_ae_loss_l1(cfg)

        if cfg.EBLL.IF_ON and "ae_l2" in cfg.EBLL.LOSS_TYPE:
            self.loss_function_map["ae_l2"] = self.get_ae_loss_l2(cfg)

    def __str__(self):
        return "Loss Type: " + " ".join(self.loss_function_map.keys())

    def optimizer_zero_grad(self):
        pass

    def optimizer_step(self):
        pass

    def scheduler_step(self):
        pass

    def get_ae_loss(self):
        ae = AELoss()

        def loss_function(data: Data):
            return ae(data.recon_ae, data.feat_t)

        return loss_function

    def get_ae_loss_l1(self, cfg):
        ae_l1 = AELossL1(cfg.EBLL.LAMBDA)

        def loss_function(data: Data):
            return ae_l1(data.ae, data.recon_ae, data.feat_t)

        return loss_function

    def get_ae_loss_l2(self, cfg):
        ae_l2 = AELossL2(cfg.EBLL.LAMBDA)

        def loss_function(data: Data):
            return ae_l2(data.ae, data.recon_ae, data.feat_t)

        return loss_function
