import copy
import logging
import sys

sys.path.append('.')
sys.path.append('..')

from engine.ebll import train_autoencoder, fine_tune_current_model, ebll_train
from utils.component import main, TrainComponent
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
    saver.checkpoint_params['model'] = source_tr.model
    saver.load_checkpoint(is_best=True)

    autoencoder_tr = TrainComponent(cfg, autoencoder=True)
    saver.checkpoint_params['autoencoder'] = source_tr.model

    logger.info("")
    logger.info('*' * 60)
    logger.info("Start training autoencoder")
    logger.info('*' * 60)
    logger.info("")

    train_autoencoder(cfg,
                      source_train_loader,
                      source_valid,
                      source_tr,
                      autoencoder_tr,
                      saver)

    saver.best_result = 0

    train_loader, num_classes = make_train_data_loader(cfg, dataset_name[1])
    ebll_valid = make_multi_valid_data_loader(cfg, [dataset_name[1]])

    current_tr = TrainComponent(cfg, num_classes)
    saver.checkpoint_params['model'] = current_tr.model
    saver.load_checkpoint(is_best=True)
    # print(current_tr)

    logger.info("")
    logger.info('*' * 60)
    logger.info("Start fine tuning current model")
    logger.info('*' * 60)
    logger.info("")

    fine_tune_current_model(cfg,
                            train_loader,
                            ebll_valid,
                            current_tr,
                            saver)

    saver.best_result = 0
    k_s = [1, 5, 10, 15, 20, 30, 50, 100, 150]
    for k in k_s:
        logger.info("")
        logger.info('*' * 60)
        logger.info(f"Start ebll training using {0.001 * k}")
        logger.info('*' * 60)
        logger.info("")

        copy_cfg = copy.deepcopy(cfg)
        copy_cfg["CONTINUATION"]["IF_ON"] = True
        copy_cfg["EBLL"]["AE_LOSS_WEIGHT"] = 0.001 * k
        ebll_tr = TrainComponent(copy_cfg, num_classes)
        ebll_tr.model = copy.deepcopy(current_tr.model)
        saver.checkpoint_params['model'] = ebll_tr.model
        ebll_valid = make_multi_valid_data_loader(cfg, dataset_name)

        ebll_train(cfg,
                   train_loader,
                   ebll_valid,
                   source_tr,
                   ebll_tr,
                   autoencoder_tr,
                   saver)

        if "ae_dist" not in cfg.CONTINUATION.LOSS_TYPE:
            break


if __name__ == '__main__':
    cfg, saver = main(["EBLL.IF_ON", True, "TEST.IF_RE_RANKING", False])
    train(cfg, saver)
