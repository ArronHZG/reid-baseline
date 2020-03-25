# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging
from time import sleep

import torch
import torch.nn as nn
from ignite.engine import Engine

from engine.trainer import create_supervised_evaluator
from utils.reid_metric import R1_mAP, R1_mAP_reranking


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")

    # multi-dataset
    validation_evaluator_list = []
    for _, n_q in num_query:

        if cfg.TEST.IF_RE_RANKING:
            logger.info("Create evaluator for reranking")
            metrics = {"r1_mAP": R1_mAP_reranking(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}
        else:
            logger.info("Create evaluator")
            metrics = {"r1_mAP": R1_mAP(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}

        evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
        validation_evaluator_list.append(evaluator)

    for index, evaluator in enumerate(validation_evaluator_list):
        evaluator.run(val_loader[index])
        if device == 'cuda':
            torch.cuda.empty_cache()
        cmc, mAP = evaluator.state.metrics['r1_mAP']
        logger.info('-' * 60)
        logger.info('Validation Results')
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        logger.info('-' * 60)
