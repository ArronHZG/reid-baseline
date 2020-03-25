import sys

sys.path.append('.')
sys.path.append('..')

from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from tools.expand import main, TrainComponent


def test(cfg, saver):
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    tr = TrainComponent(cfg, num_classes)
    to_load = {'model': tr.model}
    saver.to_save = to_load
    saver.load_checkpoint(is_best=True)
    inference(cfg, tr.model, val_loader, num_query)


if __name__ == '__main__':
    cfg, saver = main(["TEST.IF_ON", True])
    test(cfg, saver)
