# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn, optim

from solver.ranger import Ranger


def make_optimizer(cfg, model):
    # params = []
    # for key, value in model.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     lr = cfg.OPTIMIZER.BASE_LR
    #     weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY
    #     if "bias" in key:
    #         lr = cfg.OPTIMIZER.BASE_LR * cfg.OPTIMIZER.BIAS_LR_FACTOR
    #         weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY_BIAS
    #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = None
    if cfg.OPTIMIZER.NAME == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.OPTIMIZER.BASE_LR,
                              momentum=cfg.OPTIMIZER.MOMENTUM,
                              weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
                              nesterov=True)
    elif cfg.OPTIMIZER.NAME == 'Adam':
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg.OPTIMIZER.BASE_LR,
                                weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    return optimizer


if __name__ == '__main__':
    from config import cfg
    from torchvision import models

    pass
