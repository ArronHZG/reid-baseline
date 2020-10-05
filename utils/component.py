import logging
import sys
from collections import OrderedDict

sys.path.append('.')
sys.path.append('..')

logger = logging.getLogger("reid_baseline.train")


class BaseComponent:
    def __init__(self):
        pass

    def state_dict(self):
        state_dict = OrderedDict()
        for key in self.__dict__.keys():
            comp = getattr(self, key)
            if hasattr(comp, 'state_dict'):
                state_dict[key] = comp
        return state_dict

# class TrainComponent:
#     def __init__(self, cfg, num_classes=0, autoencoder=False):
#         self.device = None
#         self.model = None
#         self.loss = None
#         self.optimizer = None
#         self.scheduler = None
#         if autoencoder:
#             self.get_autoencoder_component(cfg)
#         else:
#             self.get_component(cfg, num_classes)
#         self.apex_cuda_setting(cfg)
#
#     def __str__(self):
#         s = f"{self.model}\n{self.loss}\n{self.optimizer}\n{self.scheduler}"
#         return s
#
#     def get_autoencoder_component(self, cfg):
#         self.device = cfg.MODEL.DEVICE
#         self.model = AutoEncoder(cfg.EBLL.IN_PLANES, cfg.EBLL.CODE_SIZE)
#         self.loss = AutoEncoderLoss(cfg)
#         copy_cfg = copy.deepcopy(cfg)
#         copy_cfg['OPTIMIZER']['BASE_LR'] = copy_cfg.EBLL.OPTIMIZER_BASE_LR
#         copy_cfg['OPTIMIZER']['NAME'] = 'SGD'
#         self.optimizer = make_optimizer(copy_cfg, self.model)
#         self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
#
#     def get_component(self, cfg, num_classes):
#         self.device = cfg.MODEL.DEVICE
#         self.model = build_model(cfg, num_classes)
#         self.loss = Loss(cfg, num_classes, self.model.in_planes)
#         self.optimizer = make_optimizer(cfg, self.model)
#         self.scheduler = WarmupMultiStepLR(self.optimizer,
#                                            cfg.WARMUP.STEPS,
#                                            cfg.WARMUP.GAMMA,
#                                            cfg.WARMUP.FACTOR,
#                                            cfg.WARMUP.MAX_EPOCHS,
#                                            cfg.WARMUP.METHOD)
#
#     def apex_cuda_setting(self, cfg):
#         if cfg.APEX.IF_ON:
#             logger.info("Using apex")
#             try:
#                 import apex
#             except ImportError:
#                 raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
#             assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
#             #
#             # if cfg.APEX.IF_SYNC_BN:
#             #     logger.info("Using apex synced BN")
#             #     self.module = apex.parallel.convert_syncbn_model(self.module)
#         if self.device is 'cuda':
#             self.model = self.model.cuda()
#             if cfg.APEX.IF_ON:
#                 from apex import amp
#                 amp.register_float_function(torch, 'sigmoid')
#                 self.model, self.optimizer = amp.initialize(self.model,
#                                                             self.optimizer,
#                                                             opt_level=cfg.APEX.OPT_LEVEL,
#                                                             keep_batchnorm_fp32=None if cfg.APEX.OPT_LEVEL == 'O1' else True,
#                                                             loss_scale=cfg.APEX.LOSS_SCALE[0])
