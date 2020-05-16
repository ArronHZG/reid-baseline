import sys

sys.path.append('.')
sys.path.append('..')

from data import make_multi_valid_data_loader
from engine.inference import inference
from tools.component import main, TrainComponent


def test(cfg, saver):
    dataset_name = [cfg.DATASET.NAME]
    valid = make_multi_valid_data_loader(cfg, dataset_name, verbose=True)

    tr = TrainComponent(cfg, 0)
    to_load = {'model': tr.model}
    saver.to_save = to_load
    saver.load_checkpoint(is_best=True)
    inference(cfg, tr.model, valid)


if __name__ == '__main__':
    cfg, saver = main(["TEST.IF_ON", True])
    test(cfg, saver)
