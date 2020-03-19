# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

from tools.train import main

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger


def test(cfg, saver):
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    to_load = {'model': model}
    saver.to_save = to_load
    saver.load_checkpoint(is_best=True)
    inference(cfg, model, val_loader, num_query)


if __name__ == '__main__':
    cfg, saver = main(["TEST.IF_ON", True])
    test(cfg, saver)
