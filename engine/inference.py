import logging
import sys
import time
from collections import OrderedDict

from ignite.handlers import Checkpoint

from data import make_multi_valid_data_loader
from unit.supervisedComponent import create_supervised_evaluator, TrainComponent
from utils import main

sys.path.append('.')
sys.path.append('..')
import torch

from evaluation.metric import R1_mAP

logger = logging.getLogger("reid_baseline.eval")


# def create_autoencoder_evaluator(model, autoencoder, metrics, classify_feature=True):
#     def _inference(engine, batch):
#         model.eval()
#         with torch.no_grad():
#             img, pids, camids = batch
#             img = img.cuda()
#             data = model(img)
#             auto_data = autoencoder(data.feat_t)
#             auto_data.feat_t = auto_data.recon_ae
#             auto_data.feat_c = model.bottleneck(auto_data.feat_t)
#
#             if classify_feature:
#                 return auto_data.feat_c.to(torch.float16), pids, camids
#             else:
#                 return auto_data.feat_t.to(torch.float16), pids, camids
#
#     engine = Engine(_inference)
#
#     for name, metric in metrics.items():
#         metric.attach(engine, name)
#
#     return engine
#
#
# def create_source_evaluator(source_model, current_model, metrics, classify_feature):
#     def _inference(engine, batch):
#         source_model.eval()
#         current_model.eval()
#         with torch.no_grad():
#             img, pids, camids = batch
#             img = img.cuda()
#
#             x = current_model.base(img)
#             feat_t = current_model.GAP(x).view(x.size(0), -1)
#             feat_c = source_model.bottleneck(feat_t)
#
#             if classify_feature:
#                 return feat_c.to(torch.float16), pids, camids
#             else:
#                 return feat_t.to(torch.float16), pids, camids
#
#     engine = Engine(_inference)
#
#     for name, metric in metrics.items():
#         metric.attach(engine, name)
#
#     return engine

def eval_one_dataset(cfg, name, dataloader, n_q, evaler):
    time_start = time.time()
    metric = R1_mAP(n_q,
                    max_rank=50,
                    if_feat_norm=cfg.TEST.IF_FEAT_NORM,
                    if_re_rank=cfg.TEST.IF_RE_RANKING)
    metric.attach(evaler, 'r1_mAP')
    evaler.run(dataloader)
    torch.cuda.empty_cache()
    cmc, mAP = evaler.state.metrics['r1_mAP']
    logger.info('-' * 60)
    logger.info(f"{name} Validation Results")
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    sum_result = (mAP + cmc[0]) / 2
    logger.info(f'sum_result: {sum_result:.2%}')
    cost_time = time.time() - time_start
    logger.info(f"cost_time: {cost_time:.4f} s")
    logger.info('-' * 60)


def eval_multi_dataset(cfg, valid_dict, tr_comp: TrainComponent):
    for name, (dataloader, n_q) in valid_dict.items():
        evaler = create_supervised_evaluator(tr_comp)
        eval_one_dataset(cfg, name, dataloader, n_q, evaler)


if __name__ == '__main__':
    cfg, saver = main(["TEST.IF_ON", True])
    valid_dict = make_multi_valid_data_loader(cfg, cfg.TEST.DATASET_NAMES, verbose=True)
    checkpoint = torch.load('../run/direct/market1501/resnet50/experiment-34/model/train_checkpoint_37200.pt')
    checkpoint['model'].pop('classifier.weight')
    tr_comp = TrainComponent(cfg, 0)
    to_load = tr_comp.state_dict()
    saver.load_objects(to_load={'model': to_load['model']}, checkpoint={'model': checkpoint['model']})
    eval_multi_dataset(cfg, valid_dict, tr_comp)
