import logging
import sys
import torch

from engine.inference import inference

sys.path.append('.')
sys.path.append('..')

from engine.extract import do_extract
from engine.trainer_with_feat import do_train_with_feat
from tools.component import main, TrainComponent
from data import make_multi_valid_data_loader, \
    make_data_with_loader_with_feat_label, make_train_data_loader_for_extract

logger = logging.getLogger("reid_baseline.feat")


def train(cfg, saver):
    """
        Train a new dataset with source data
        e.g.: train: dukemtmc with market feat
    """

    source_name = cfg.DATASET.NAME
    target_name = cfg.FEAT.DATASET_NAME

    source_loader, _ = make_train_data_loader_for_extract(cfg, source_name)

    tr = TrainComponent(cfg, 702)
    to_load = {'model': tr.model}
    saver.checkpoint_params = to_load
    saver.load_checkpoint(is_best=True)

    feat, _ = do_extract(cfg, source_loader, tr)
    feat = feat.cpu()
    if cfg.MODEL.DEVICE == 'cuda':
        torch.cuda.empty_cache()
    logger.info(f"Extracting feat is done. {feat.size()}")

    train_loader, num_classes = make_data_with_loader_with_feat_label(cfg, source_name, target_name, feat)

    valid = make_multi_valid_data_loader(cfg, [source_name])

    # inference(cfg, train_component.module, valid)

    do_train_with_feat(cfg,
                       train_loader,
                       valid,
                       tr,
                       saver)


if __name__ == '__main__':
    cfg, saver = main(["FEAT.IF_ON", True])
    train(cfg, saver)
