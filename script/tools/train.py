import sys

sys.path.append('.')
sys.path.append('..')

from data import make_data_loader
from engine.trainer import do_train
from script.tools.expand import TrainComponent, main


def train(cfg, saver):
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    train_component = TrainComponent(cfg, num_classes)

    do_train(
        cfg,
        train_component.model,
        train_loader,
        val_loader,
        train_component.optimizer,
        train_component.scheduler,
        train_component.loss_function,
        num_query,
        saver,
        center_criterion=train_component.loss_center,
        optimizer_center=train_component.optimizer_center)


if __name__ == '__main__':
    cfg, saver = main()
    train(cfg, saver)
