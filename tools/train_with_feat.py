import sys

from engine.extract import do_extract

sys.path.append('.')
sys.path.append('..')

from tools.expand import main, TrainComponent
from data import make_multi_valid_data_loader, \
    make_data_with_loader_with_feat, make_train_data_loader_for_extract
from engine.trainer import do_train


def train(cfg, saver):
    """
        Train a new dataset with source data
        e.g.: train: dukemtmc with market feat
    """

    source_name = cfg.DATASETS.NAME
    target_name = cfg.FEAT.DATASETS_NAME

    source_loader, _ = make_train_data_loader_for_extract(cfg, source_name)

    tr = TrainComponent(cfg, 0)
    to_load = {'model': tr.model}
    saver.to_save = to_load
    saver.load_checkpoint(is_best=True)

    feat, _ = do_extract(cfg, source_loader, tr)
    del tr

    train_loader, num_classes = make_data_with_loader_with_feat(cfg, source_name, target_name, feat)

    valid = make_multi_valid_data_loader(cfg, [source_name, target_name])

    train_component = TrainComponent(cfg, num_classes)

    do_train(cfg,
             train_loader,
             valid,
             train_component,
             saver)


if __name__ == '__main__':
    cfg, saver = main(["FEAT.IF_ON", True])
    train(cfg, saver)
