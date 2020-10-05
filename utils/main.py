import argparse
import logging
import os
import random

import torch
import numpy as np
from torch.backends import cudnn

from config import cfg
from utils.logger import setup_logger
from utils.saver import Saver


def main(merge_list=None):
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if merge_list:
        cfg.merge_from_list(merge_list)

    saver = Saver(cfg)
    if cfg.TEST.IF_ON:
        log_file = 'test-log.txt'
    else:
        log_file = 'train-log.txt'
    logger = setup_logger("reid_baseline", saver.save_dir, log_file)
    logger.setLevel(logging.INFO)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))

    logger.info("Running with config:\n{}".format(cfg))
    logger.info("=" * 20)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.cuda.set_device(cfg.MODEL.DEVICE_ID)
    logger.info(f"Using GPU: {cfg.MODEL.DEVICE_ID}")
    logger.info(f"CUDNN VERSION: {cudnn.version()}")

    cudnn.enabled = True
    cudnn.benchmark = True
    if cfg.MODEL.IF_DETERMINISTIC:
        # using cuDNN
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.random.manual_seed(1024)
        torch.random.manual_seed(1024)
        torch.cuda.manual_seed(1024)  # gpu
        torch.cuda.manual_seed_all(1024)
        np.random.seed(1024)  # numpy
        random.seed(1024)  # random and transforms
        torch.set_printoptions(precision=10)

    return cfg, saver
