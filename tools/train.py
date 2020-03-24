import sys

from tools.expand import main, TrainComponent

sys.path.append('.')
sys.path.append('..')

from data import make_data_loader
from engine.trainer import do_train


def train(cfg, saver):
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    train_component = TrainComponent(cfg, num_classes)

    do_train(cfg,
             train_loader,
             val_loader,
             train_component,
             num_query,
             saver)


if __name__ == '__main__':
    cfg, saver = main()
    train(cfg, saver)
