import logging
import sys

sys.path.append('.')
sys.path.append('..')
from engine.uda import do_uda
from tools.expand import main, TrainComponent
from data import make_data_loader


def cluster(cfg, saver):
    do_uda(cfg, saver)


if __name__ == '__main__':
    cfg, saver = main(["TEST.IF_ON", True, "UDA.IF_ON", True])
    cluster(cfg, saver)
