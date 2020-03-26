import sys

sys.path.append('.')
sys.path.append('..')

from tools.expand import main, TrainComponent
from data import make_train_data_loader, make_train_data_loader_with_expand, make_multi_valid_data_loader, \
    make_data_with_loader_with_feat
from engine.trainer import do_train


def train(cfg, saver):
    '''
    Train a new dataset with source data
    e.g.: train: dukemtmc with market feat

    :param cfg:
    :param saver:
    :return:
    '''
    if len(cfg.DATASETS.EXPAND) != 1:
        return
    feat_name = cfg.DATASETS.EXPAND[0]
    dataset_name = [cfg.DATASETS.NAME, feat_name]




    train_loader, num_classes = make_data_with_loader_with_feat(cfg, dataset_name, feat)

    valid = make_multi_valid_data_loader(cfg, dataset_name)

    train_component = TrainComponent(cfg, num_classes)

    do_train(cfg,
             train_loader,
             valid,
             train_component,
             saver)


if __name__ == '__main__':
    cfg, saver = main()
    train(cfg, saver)
