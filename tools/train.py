import argparse
import logging
import os
import random
import sys
import numpy as np

import torch
from torch.backends import cudnn

from utils.saver import Saver

sys.path.append('.')
sys.path.append('..')

from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from loss import Loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger


def train(cfg, saver):
    logger = logging.getLogger("reid_baseline")

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    model = build_model(cfg, num_classes)

    loss = Loss(cfg, num_classes)
    loss_function = loss.make_loss()

    optimizer, optimizer_center = make_optimizer(cfg, model, loss.center)

    scheduler = WarmupMultiStepLR(optimizer,
                                  cfg.WARMUP.STEPS,
                                  cfg.WARMUP.GAMMA,
                                  cfg.WARMUP.FACTOR,
                                  cfg.WARMUP.MAX_EPOCHS,
                                  cfg.WARMUP.METHOD)

    start_epoch = 0

    if cfg.APEX.IF_ON:
        logger.info("Using apex")
        try:
            import apex
            # from apex.parallel import DistributedDataParallel as DDP
            # from apex.fp16_utils import *
            # from apex import amp, optimizers
            # from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

        if cfg.APEX.IF_SYNC_BN:
            logger.info("Using apex synced BN")
            model = apex.parallel.convert_syncbn_model(model)

    # device = cfg.MODEL.DEVICE
    if cfg.MODEL.DEVICE is 'cuda':
        model = model.cuda()
        if cfg.APEX.IF_ON:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level=cfg.APEX.OPT_LEVEL,
                                              keep_batchnorm_fp32=None if cfg.APEX.OPT_LEVEL == 'O1' else True,
                                              loss_scale=cfg.APEX.LOSS_SCALE[0])

            if optimizer_center:
                loss.center, optimizer_center = amp.initialize(loss.center, optimizer_center,
                                                               opt_level=cfg.APEX.OPT_LEVEL,
                                                               keep_batchnorm_fp32=None if cfg.APEX.OPT_LEVEL == 'O1' else True,
                                                               loss_scale=cfg.APEX.LOSS_SCALE[0])

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,  # modify for using self trained model
        loss_function,
        num_query,
        start_epoch,  # add for using self trained model
        saver,
        center_criterion=loss.center,
        optimizer_center=optimizer_center)


def main():
    # parser = argparse.ArgumentParser(description="ReID Baseline Training")
    #
    # parser.add_argument("--config_file", default="", help="path to config file", type=str)
    # parser.add_argument("opts", help="Modify config options using the command-line",
    #                     default=None, nargs=argparse.REMAINDER)
    #
    # args = parser.parse_args()
    #
    #
    # #
    # if args.config_file != "":
    #     cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()
    # cfg.merge_from_file("/home/arron/PycharmProjects/reid-baseline/configs/softmax_triplet_with_center.yml")
    cfg.merge_from_file("/home/arron/PycharmProjects/reid-baseline/configs/apex.yml")
    cfg.freeze()

    saver = Saver(cfg)
    logger = setup_logger("reid_baseline", saver.save_path, 0)
    logger.setLevel(logging.INFO)

    # logger.info("Using {} GPUS".format(num_gpus))
    # logger.info(args)
    #
    # if args.config_file != "":
    #     logger.info("Loaded configuration file {}".format(args.config_file))
    #     with open(args.config_file, 'r') as cf:
    #         config_str = "\n" + cf.read()
    #         logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("=" * 20)
    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        logger.info(f"Using GPU: {cfg.MODEL.DEVICE_ID}")
        logger.info(f"CUDNN VERSION: {cudnn.version()}")

    cudnn.benchmark = True
    if cfg.MODEL.IF_DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(1024)
        torch.cuda.manual_seed(1024)  # gpu
        np.random.seed(1024)  # numpy
        random.seed(1024)  # random and transforms
        torch.set_printoptions(precision=10)

    train(cfg, saver)


if __name__ == '__main__':
    main()

# args.distributed = False
#     if 'WORLD_SIZE' in os.environ:
#         args.distributed = int(os.environ['WORLD_SIZE']) > 1
#
#     args.gpu = 0
#     args.world_size = 1
#
#     if args.distributed:
#         args.gpu = args.local_rank
#         torch.cuda.set_device(args.gpu)
#         torch.distributed.init_process_group(backend='nccl',
#                                              init_method='env://')
#         args.world_size = torch.distributed.get_world_size()

# if args.distributed:
#     # By default, apex.parallel.DistributedDataParallel overlaps communication with
#     # computation in the backward pass.
#     # model = DDP(model)
#     # delay_allreduce delays all communication to the end of the backward pass.
#     model = DDP(model, delay_allreduce=True)
# if device:
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)
#     model.to(device)
