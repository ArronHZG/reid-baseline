import logging
import sys

sys.path.append('.')
sys.path.append('..')

from tools.component import main, TrainComponent
from engine.continuous_trainer import do_continuous_train
from engine.inference import inference
from data import make_multi_valid_data_loader, make_train_data_loader

logger = logging.getLogger("reid_baseline.continuation")


def train(cfg, saver):
    """
        Train a new dataset with distillation
        e.g.: train: dukemtmc with market module
    """
    source_tr = TrainComponent(cfg, 0)
    to_load = {'module': source_tr.model}
    saver.to_save = to_load
    saver.load_checkpoint(is_best=True)

    dataset_name = [cfg.DATASET.NAME, cfg.CONTINUATION.DATASET_NAME]

    train_loader, num_classes = make_train_data_loader(cfg, dataset_name[1])

    current_tr = TrainComponent(cfg, num_classes)
    to_load = {'module': current_tr.model}
    saver.to_save = to_load
    saver.load_checkpoint(is_best=True)

    valid = make_multi_valid_data_loader(cfg, dataset_name)

    # inference(cfg, current_tr.module, valid)

    do_continuous_train(cfg,
                        train_loader,
                        valid,
                        source_tr,
                        current_tr,
                        saver)


if __name__ == '__main__':
    cfg, saver = main(["CONTINUATION.IF_ON", True, "TEST.IF_RE_RANKING", False])
    train(cfg, saver)
