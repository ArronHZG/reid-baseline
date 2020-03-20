import logging
import sys

sys.path.append('.')
sys.path.append('..')
from engine.cluster import do_cluster
from tools.expand import main, TrainComponent
from data import make_data_loader


def cluster(cfg, saver):
    do_cluster(cfg, saver)


if __name__ == '__main__':
    cfg, saver = main(["TEST.IF_ON", True, "CLUSTER.IF_ON", True])
    cluster(cfg, saver)
