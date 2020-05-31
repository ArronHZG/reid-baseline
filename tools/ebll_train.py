import copy
import logging
import sys

sys.path.append('.')
sys.path.append('..')

from engine.ebll import train_autoencoder, fine_tune_current_model, ebll_train
from tools.component import main, TrainComponent
from data import make_multi_valid_data_loader, make_train_data_loader

logger = logging.getLogger("reid_baseline.continuation")


def train(cfg, saver):
    """
        Train a new dataset with distillation
        e.g.: train: dukemtmc with market module
    """
    dataset_name = [cfg.DATASET.NAME, cfg.EBLL.DATASET_NAME]
    source_train_loader, source_num_classes = make_train_data_loader(cfg, dataset_name[0])
    source_valid = make_multi_valid_data_loader(cfg, [dataset_name[0]])

    source_tr = TrainComponent(cfg)
    saver.to_save = {'model': source_tr.model}
    saver.load_checkpoint(is_best=True)

    autoencoder_tr = TrainComponent(cfg, autoencoder=True)
    saver.to_save = {'autoencoder': source_tr.model}

    train_autoencoder(cfg,
                      source_train_loader,
                      source_valid,
                      source_tr,
                      autoencoder_tr,
                      saver)

    train_loader, num_classes = make_train_data_loader(cfg, dataset_name[1])
    ebll_valid = make_multi_valid_data_loader(cfg, [dataset_name[1]])

    current_tr = TrainComponent(cfg, num_classes)
    saver.to_save = {'model': current_tr.model}
    saver.load_checkpoint(is_best=True)
    # print(current_tr)

    fine_tune_current_model(cfg,
                            train_loader,
                            ebll_valid,
                            current_tr,
                            saver)

    copy_cfg = copy.deepcopy(cfg)
    copy_cfg["CONTINUATION"]["IF_ON"] = True
    ebll_tr = TrainComponent(copy_cfg, num_classes)
    ebll_tr.model = current_tr.model

    ebll_valid = make_multi_valid_data_loader(cfg, dataset_name)

    ebll_train(cfg,
               train_loader,
               ebll_valid,
               source_tr,
               ebll_tr,
               autoencoder_tr,
               saver)


if __name__ == '__main__':
    cfg, saver = main(["EBLL.IF_ON", True, "TEST.IF_RE_RANKING", False])
    train(cfg, saver)
