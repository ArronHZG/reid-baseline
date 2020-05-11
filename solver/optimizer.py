# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch

from solver.ranger import Ranger


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.OPTIMIZER.BASE_LR
        weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.OPTIMIZER.BASE_LR * cfg.OPTIMIZER.BIAS_LR_FACTOR
            weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.OPTIMIZER.NAME == 'ranger':
        # optimizer = Ranger(params,
        #                    lr=cfg.OPTIMIZER.BASE_LR,
        #                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        #                    N_sma_threshhold=5,  # Ranger options
        #                    betas=(.95, 0.999))

        optimizer = Ranger(params,
                           lr=cfg.OPTIMIZER.BASE_LR,
                           weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
                           N_sma_threshhold=4,  # Ranger options
                           betas=(.90, 0.999))

    elif cfg.OPTIMIZER.NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.OPTIMIZER.NAME)(params,
                                                             momentum=cfg.SOLVER.MOMENTUM,
                                                             nesterov=True)
    else:
        optimizer = getattr(torch.optim, cfg.OPTIMIZER.NAME)(params)
    return optimizer


if __name__ == '__main__':
    from config import cfg
    from torchvision import models

    pass
