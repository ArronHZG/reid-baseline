from collections import OrderedDict

import torch
from apex import amp
from ignite.engine import Engine
from torch import optim
from yacs.config import CfgNode

from loss import MyCrossEntropy, ArcfaceLoss, TripletLoss, CenterLoss
from modeling import build_model
from solver import WarmupMultiStepLR
from utils import BaseComponent, Data


class TrainComponent(BaseComponent):
    def __init__(self, cfg: CfgNode, num_classes):
        super().__init__()
        self.model = build_model(cfg, num_classes).cuda()

        # loss
        self.loss_type = cfg.LOSS.LOSS_TYPE
        self.loss_function_map = OrderedDict()

        # ID loss
        if 'softmax' in self.loss_type:
            self.xent = MyCrossEntropy(num_classes=num_classes,
                                       label_smooth=cfg.LOSS.IF_LABEL_SMOOTH,
                                       learning_weight=cfg.LOSS.IF_LEARNING_WEIGHT).cuda()

            def loss_function(data: Data):
                return cfg.LOSS.ID_LOSS_WEIGHT * self.xent(data.cls_score, data.cls_label)

            self.loss_function_map["softmax"] = loss_function

        if 'arcface' in self.loss_type:
            self.arcface = ArcfaceLoss(num_classes=num_classes, feat_dim=self.model.in_planes).cuda()

            def loss_function(data: Data):
                return self.arcface(data.feat_c, data.cls_label)

            self.loss_function_map["arcface"] = loss_function

        # metric loss
        if 'triplet' in self.loss_type:
            self.triplet = TripletLoss(cfg.LOSS.MARGIN,
                                       learning_weight=False).cuda()

            def loss_function(data: Data):
                return cfg.LOSS.METRIC_LOSS_WEIGHT * self.triplet(data.feat_t, data.cls_label)

            self.loss_function_map["triplet"] = loss_function

        # cluster loss
        if 'center' in self.loss_type:
            self.center = CenterLoss(num_classes=num_classes,
                                     feat_dim=self.model.in_planes,
                                     loss_weight=cfg.LOSS.CENTER_LOSS_WEIGHT,
                                     learning_weight=False).cuda()

            def loss_function(data: Data):
                return cfg.LOSS.CENTER_LOSS_WEIGHT * self.center(data.feat_t, data.cls_label)

            self.loss_function_map["center"] = loss_function

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=cfg.OPTIMIZER.BASE_LR,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

        self.xent_optimizer = optim.SGD(self.xent.parameters(),
                                        lr=cfg.OPTIMIZER.LOSS_LR / 5,
                                        momentum=cfg.OPTIMIZER.MOMENTUM,
                                        weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

        self.center_optimizer = optim.SGD(self.center.parameters(),
                                          lr=cfg.OPTIMIZER.LOSS_LR,
                                          momentum=cfg.OPTIMIZER.MOMENTUM,
                                          weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

        model_list = [self.model, self.xent, self.triplet, self.center]
        opt_list = [self.optimizer, self.center_optimizer, self.xent_optimizer]
        model_list, opt_list = amp.initialize(model_list,
                                              opt_list,
                                              opt_level='O1',
                                              loss_scale=1.0)
        self.model, self.xent, self.triplet, self.center = model_list[0], model_list[1], model_list[2], model_list[3]
        self.optimizer, self.center_optimizer, self.xent_optimizer = opt_list[0], opt_list[1], opt_list[2]

        self.scheduler = WarmupMultiStepLR(self.optimizer,
                                           cfg.WARMUP.STEPS,
                                           cfg.WARMUP.GAMMA,
                                           cfg.WARMUP.FACTOR,
                                           cfg.WARMUP.MAX_EPOCHS,
                                           cfg.WARMUP.METHOD)

    def __str__(self):
        s = f"{self.model}\n{self.loss_function_map.keys()}\n{self.optimizer}\n{self.scheduler}"
        return s


def create_supervised_trainer(tr_comp: TrainComponent):
    def _update(engine, batch):
        tr_comp.model.train()
        tr_comp.optimizer.zero_grad()
        tr_comp.center_optimizer.zero_grad()
        tr_comp.xent_optimizer.zero_grad()

        img, cls_label = batch
        img = img.to(torch.float).cuda()
        cls_label = cls_label.cuda()
        data = tr_comp.model(img)
        data.cls_label = cls_label

        loss_values = {}
        loss = torch.tensor(0.0, requires_grad=True).cuda()
        for name, loss_fn in tr_comp.loss_function_map.items():
            loss_temp = loss_fn(data)
            loss += loss_temp
            loss_values[name] = loss_temp.item()

        with amp.scale_loss(loss, tr_comp.optimizer) as scaled_loss:
            scaled_loss.backward()

        if 'center' in tr_comp.loss_type:
            for param in tr_comp.center.parameters():
                param.grad.data *= (1. / tr_comp.center.loss_weight)
        tr_comp.optimizer.step()
        tr_comp.center_optimizer.step()
        tr_comp.xent_optimizer.step()

        # compute acc
        acc = (data.cls_score.max(1)[1] == data.cls_label).float().mean()
        loss_values["Loss"] = loss.item()
        loss_values["Acc"] = acc.item()
        return loss_values

    return Engine(_update)


def create_supervised_evaluator(tr_comp: TrainComponent, classify_feature=True):
    def _inference(engine, batch):
        tr_comp.model.eval()
        with torch.no_grad():
            img, pids, camids = batch
            img = img.cuda()
            data = tr_comp.model(img)

            if classify_feature:
                return data.feat_c.to(torch.float16), pids, camids
            else:
                return data.feat_t.to(torch.float16), pids, camids

    return Engine(_inference)

