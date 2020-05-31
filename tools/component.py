import argparse
import copy
import logging
import os
import random
import sys
import numpy as np
import torch
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR

sys.path.append('.')
sys.path.append('..')

from config import cfg
from loss import Loss, AutoEncoderLoss
from modeling import build_model, AutoEncoder
from solver import make_optimizer, WarmupMultiStepLR
from utils.logger import setup_logger
from utils.saver import Saver

logger = logging.getLogger("reid_baseline.train")


class TrainComponent:
    def __init__(self, cfg, num_classes=0, autoencoder=False):
        self.device = None
        self.model = None
        self.loss = None
        self.optimizer = None
        self.scheduler = None
        if autoencoder:
            self.get_autoencoder_component(cfg)
        else:
            self.get_component(cfg, num_classes)
        self.apex_cuda_setting(cfg)

    def __str__(self):
        s = f"{self.model}\n{self.loss}\n{self.optimizer}\n{self.scheduler}"
        return s

    def get_autoencoder_component(self, cfg):
        self.device = cfg.MODEL.DEVICE
        self.model = AutoEncoder(cfg.EBLL.IN_PLANES, cfg.EBLL.CODE_SIZE)
        self.loss = AutoEncoderLoss(cfg)
        copy_cfg = copy.deepcopy(cfg)
        copy_cfg['OPTIMIZER']['BASE_LR'] = copy_cfg.EBLL.OPTIMIZER_BASE_LR
        copy_cfg['OPTIMIZER']['NAME'] = 'SGD'
        self.optimizer = make_optimizer(copy_cfg, self.model)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

    def get_component(self, cfg, num_classes):
        self.device = cfg.MODEL.DEVICE
        self.model = build_model(cfg, num_classes)
        self.loss = Loss(cfg, num_classes, self.model.in_planes)
        self.optimizer = make_optimizer(cfg, self.model)
        self.scheduler = WarmupMultiStepLR(self.optimizer,
                                           cfg.WARMUP.STEPS,
                                           cfg.WARMUP.GAMMA,
                                           cfg.WARMUP.FACTOR,
                                           cfg.WARMUP.MAX_EPOCHS,
                                           cfg.WARMUP.METHOD)

    def apex_cuda_setting(self, cfg):
        if cfg.APEX.IF_ON:
            logger.info("Using apex")
            try:
                import apex
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
            assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
            #
            # if cfg.APEX.IF_SYNC_BN:
            #     logger.info("Using apex synced BN")
            #     self.module = apex.parallel.convert_syncbn_model(self.module)
        if self.device is 'cuda':
            self.model = self.model.cuda()
            if cfg.APEX.IF_ON:
                from apex import amp
                amp.register_float_function(torch, 'sigmoid')
                self.model, self.optimizer = amp.initialize(self.model,
                                                            self.optimizer,
                                                            opt_level=cfg.APEX.OPT_LEVEL,
                                                            keep_batchnorm_fp32=None if cfg.APEX.OPT_LEVEL == 'O1' else True,
                                                            loss_scale=cfg.APEX.LOSS_SCALE[0])


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
    logger = setup_logger("reid_baseline", saver.save_dir, 0)
    logger.setLevel(logging.INFO)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    #     with open(args.config_file, 'r') as cf:
    #         config_str = "\n" + cf.read()
    #         logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    logger.info("=" * 20)
    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg.MODEL.DEVICE_ID}'
        # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        logger.info(f"Using GPU: {cfg.MODEL.DEVICE_ID}")
        logger.info(f"CUDNN VERSION: {cudnn.version()}")

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
#     # module = DDP(module)
#     # delay_allreduce delays all communication to the end of the backward pass.
#     module = DDP(module, delay_allreduce=True)
# if device:
#     if torch.cuda.device_count() > 1:
#         module = nn.DataParallel(module)
#     module.to(device)
