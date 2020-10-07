import logging
from collections import OrderedDict

import torch
from ignite.engine import Engine

from evaluation.metric import R1_mAP, R1_mAP_reranking

logger = logging.getLogger("reid_baseline.eval")


def create_supervised_evaluator(model, metrics, classify_feature=True):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            img, pids, camids = batch
            img = img.cuda()
            data = model(img)

            if classify_feature:
                return data.feat_c.to(torch.float16), pids, camids
            else:
                return data.feat_t.to(torch.float16), pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_autoencoder_evaluator(model, autoencoder, metrics, classify_feature=True):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            img, pids, camids = batch
            img = img.cuda()
            data = model(img)
            auto_data = autoencoder(data.feat_t)
            auto_data.feat_t = auto_data.recon_ae
            auto_data.feat_c = model.bottleneck(auto_data.feat_t)

            if classify_feature:
                return auto_data.feat_c.to(torch.float16), pids, camids
            else:
                return auto_data.feat_t.to(torch.float16), pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_source_evaluator(source_model, current_model, metrics, classify_feature):
    def _inference(engine, batch):
        source_model.eval()
        current_model.eval()
        with torch.no_grad():
            img, pids, camids = batch
            img = img.cuda()

            x = current_model.base(img)
            feat_t = current_model.GAP(x).view(x.size(0), -1)
            feat_c = source_model.bottleneck(feat_t)

            if classify_feature:
                return feat_c.to(torch.float16), pids, camids
            else:
                return feat_t.to(torch.float16), pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class Eval:
    def __init__(self, valid_dict, re_ranking=False, classify_feature=True):
        self.validation_evaluator_map = None
        self.valid_dict = valid_dict
        self.re_ranking = re_ranking
        self.classify_feature = classify_feature

    def eval_multi_dataset(self):
        logger.info('-' * 60)
        sum_result = 0
        for name, validation_evaluator in self.validation_evaluator_map.items():
            validation_evaluator.run(self.valid_dict[name][0])
            torch.cuda.empty_cache()
            cmc, mAP = validation_evaluator.state.metrics['r1_mAP']
            logger.info(f"{name} Validation Results")
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            sum_result += (mAP + cmc[0]) / 2
        sum_result /= len(self.validation_evaluator_map)
        logger.info('-' * 60)
        return sum_result

    def get_valid_eval_map(self, cfg, model):
        self.validation_evaluator_map = OrderedDict()
        for name, (_, n_q) in self.valid_dict.items():
            if self.re_ranking:
                metrics = {"r1_mAP": R1_mAP_reranking(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}
            else:
                metrics = {"r1_mAP": R1_mAP(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}

            self.validation_evaluator_map[name] = create_supervised_evaluator(model,
                                                                              metrics=metrics,
                                                                              classify_feature=self.classify_feature)

    def get_valid_eval_map_autoencoder(self, cfg, model, autoencoder):
        self.validation_evaluator_map = OrderedDict()
        for name, (_, n_q) in self.valid_dict.items():
            if self.re_ranking:
                metrics = {"r1_mAP": R1_mAP_reranking(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}
            else:
                metrics = {"r1_mAP": R1_mAP(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}

            self.validation_evaluator_map[name] = create_autoencoder_evaluator(model,
                                                                               autoencoder,
                                                                               metrics=metrics,
                                                                               classify_feature=self.classify_feature)

    def get_valid_eval_map_ebll(self, cfg, source_model, current_model):
        self.validation_evaluator_map = OrderedDict()
        long = len(self.valid_dict.items())
        list_odict_items = list(self.valid_dict.items())

        for i in range(long - 1):
            name, (_, n_q) = list_odict_items[i]
            if self.re_ranking:
                metrics = {"r1_mAP": R1_mAP_reranking(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}
            else:
                metrics = {"r1_mAP": R1_mAP(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}

            self.validation_evaluator_map[name] = create_source_evaluator(source_model,
                                                                          current_model,
                                                                          metrics=metrics,
                                                                          classify_feature=self.classify_feature)

        name, (_, n_q) = list_odict_items[long - 1]
        if self.re_ranking:
            metrics = {"r1_mAP": R1_mAP_reranking(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}
        else:
            metrics = {"r1_mAP": R1_mAP(n_q, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}

        self.validation_evaluator_map[name] = create_supervised_evaluator(current_model,
                                                                          metrics=metrics,
                                                                          classify_feature=self.classify_feature)


def inference(
        cfg,
        model,
        valid
):
    # multi-dataset
    ev = Eval(valid, cfg.TEST.IF_RE_RANKING, cfg.TEST.IF_CLASSIFT_FEATURE)
    ev.get_valid_eval_map(cfg, model)
    sum_result = ev.eval_multi_dataset()
    logger.info(f'Sum result: {sum_result:.4f}')
