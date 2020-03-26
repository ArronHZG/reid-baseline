import sys


sys.path.append('.')
sys.path.append('..')

from data import make_train_data_loader, make_train_data_loader_with_expand, make_multi_valid_data_loader
from tools.expand import main, TrainComponent
from engine.trainer import do_train


def train(cfg, saver):
    dataset_name = [cfg.DATASETS.NAME]

    for name in cfg.DATASETS.EXPAND:
        dataset_name.append(name)
    if len(dataset_name) == 1:
        train_loader, num_classes = make_train_data_loader(cfg, dataset_name[0])
    else:
        train_loader, num_classes = make_train_data_loader_with_expand(cfg, dataset_name)

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
