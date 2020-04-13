import logging
from collections import OrderedDict

from torch import nn

from .center_loss import CenterLoss
from .dec_loss import DECLoss
from .smoth_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .dist_loss import CrossEntropyDistLoss

logger = logging.getLogger("reid_baseline.loss")


class Loss:
    def __init__(self, cfg, num_classes, feat_dim):  # modified by arron

        self.cfg = cfg
        self.num_classes = num_classes
        self.loss_type = cfg.LOSS.LOSS_TYPE

        # ID loss
        if self.cfg.LOSS.IF_LABEL_SMOOTH:
            self.xent = CrossEntropyLabelSmooth(num_classes=self.num_classes)
            logger.info(f"Label smooth on, numclasses: {self.num_classes}")
        else:
            self.xent = nn.CrossEntropyLoss()

        # m loss
        self.triplet = TripletLoss(self.cfg.LOSS.MARGIN)

        # cluster loss
        self.center_loss_weight = cfg.LOSS.CENTER_LOSS_WEIGHT
        self.center = CenterLoss(num_classes=self.num_classes, feat_dim=feat_dim)

        self.dec = DECLoss()

        self.loss_function_map = OrderedDict()
        self.make_loss_map()
        self.cross_entropy_dist_loss = CrossEntropyDistLoss(T=cfg.CONTINUATION.T)

    def make_loss_map(self):
        """
        **kw:
            feat_t,
            feat_c,
            cls_score,
            target,
            target_feat_c,
        :return:
        """

        if 'softmax' in self.loss_type:
            def loss_function(**kw):
                return self.xent(kw['cls_score'], kw['target'])

            self.loss_function_map["softmax"] = loss_function

        if 'triplet' in self.loss_type:
            def loss_function(**kw):
                return self.triplet(kw['feat_t'], kw['target'])

            self.loss_function_map["triplet"] = loss_function

        if self.cfg.LOSS.IF_WITH_CENTER:
            def loss_function(**kw):
                return self.center_loss_weight * self.center(kw['feat_t'], kw['target'])

            self.loss_function_map["center"] = loss_function

        if self.cfg.LOSS.IF_WITH_CENTER and self.cfg.LOSS.IF_WITH_DEC:
            def loss_function(**kw):
                return self.dec(kw['feat_t'], self.center.centers)

            self.loss_function_map["dec"] = loss_function

        if self.cfg.CONTINUATION.IF_ON:
            def loss_function(**kw):
                return self.cross_entropy_dist_loss(kw['feat_c'], kw['target_feat_c'])

            self.loss_function_map["dist"] = loss_function
