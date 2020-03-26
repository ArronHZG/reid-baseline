# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging
from collections import OrderedDict

import torch
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_reranking
from utils.tensor_utils import batch_horizontal_flip

logger = logging.getLogger("reid_baseline.eval")


def get_valid_eval_map(cfg, device, model, valid, re_ranking=False):
    validation_evaluator_map = OrderedDict()
    for name, (_, n_q) in valid.items():
        if re_ranking:
            metrics = {"r1_mAP": R1_mAP_reranking(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}
        else:
            metrics = {"r1_mAP": R1_mAP(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}

        validation_evaluator_map[name] = create_supervised_evaluator(model, metrics=metrics, device=device)
    return validation_evaluator_map


def create_supervised_evaluator(model, metrics, device, flip=False):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            img, pids, camids = batch
            img = img.to(device)
            feat = model(img).to(torch.float16)
            if flip:
                flip_img = batch_horizontal_flip(img, device)
                flip_feat = model(flip_img).to(torch.float16)
                feat += flip_feat
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def eval_multi_dataset(device, validation_evaluator_map, valid):
    logger.info('-' * 60)
    sum_result = 0
    for name, validation_evaluator in validation_evaluator_map.items():
        validation_evaluator.run(valid[name][0])
        if device == 'cuda':
            torch.cuda.empty_cache()

        cmc, mAP = validation_evaluator.state.metrics['r1_mAP']
        logger.info(f"{name} Validation Results")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        sum_result += (mAP + cmc[0]) / 2
    sum_result /= len(validation_evaluator_map)
    logger.info('-' * 60)
    return sum_result


def inference(
        cfg,
        model,
        valid
):
    device = cfg.MODEL.DEVICE
    # multi-dataset
    validation_evaluator_map = get_valid_eval_map(cfg, device, model, valid, cfg.TEST.IF_RE_RANKING)
    eval_multi_dataset(device, validation_evaluator_map, valid)
