import sys

sys.path.append('.')
sys.path.append('..')

from data import make_multi_valid_data_loader
from engine.inference import inference
from utils.component import main, TrainComponent


def test(cfg, saver):
    dataset_name = [cfg.DATASET.NAME]
    valid = make_multi_valid_data_loader(cfg, dataset_name, verbose=True)

    tr = TrainComponent(cfg)
    saver.checkpoint_params['model'] = tr.model
    saver.load_checkpoint(is_best=True)
    inference(cfg, tr.model, valid)


if __name__ == '__main__':
    cfg, saver = main(["TEST.IF_ON", True])
    test(cfg, saver)
