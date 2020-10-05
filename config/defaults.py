from yacs.config import CfgNode

cfg = CfgNode()

cfg.DATASET = CfgNode()
cfg.DATASET.NAME = 'market1501'  # market1501, dukemtmc, msmt17
cfg.DATASET.ROOT_DIR = '/home/arron/dataset'

cfg.DATALOADER = CfgNode()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.DATALOADER.NUM_INSTANCE = 4
cfg.DATALOADER.SAMPLER = 'RandomIdentity'  # Options: 'None' or 'RandomIdentity'

cfg.INPUT = CfgNode()
cfg.INPUT.SIZE_TRAIN = [256, 128]
cfg.INPUT.SIZE_TEST = [256, 128]
cfg.INPUT.PROB = 0.5
cfg.INPUT.RE_PROB = 0.5
cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
cfg.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
cfg.INPUT.PADDING = 10

cfg.MODEL = CfgNode()
cfg.MODEL.DEVICE_ID = 1
cfg.MODEL.IF_DETERMINISTIC = False
cfg.MODEL.NAME = 'resnet50'
cfg.MODEL.LAST_STRIDE = 1
cfg.MODEL.PRETRAIN_CHOICE = 'imagenet'  # Options: 'imagenet','random' or 'self'
cfg.MODEL.IF_IBN_A = True
cfg.MODEL.IF_IBN_B = False
cfg.MODEL.IF_SE = False

cfg.LOSS = CfgNode()
# Options: 'softmax' 'triplet' 'softmax_triplet' 'softmax_arcface_triplet'
cfg.LOSS.LOSS_TYPE = 'softmax_triplet_center'
cfg.LOSS.IF_LABEL_SMOOTH = True
cfg.LOSS.IF_WITH_CENTER = True
cfg.LOSS.CENTER_LOSS_WEIGHT = 0.0005
cfg.LOSS.IF_WITH_DEC = False
cfg.LOSS.METRIC_LOSS_WEIGHT = 1.0
cfg.LOSS.ID_LOSS_WEIGHT = 1.0
cfg.LOSS.MARGIN = 0.3
cfg.LOSS.IF_LEARNING_WEIGHT = True

cfg.OPTIMIZER = CfgNode()
cfg.OPTIMIZER.BASE_LR = 0.00035
cfg.OPTIMIZER.NAME = 'Adam'  # Adam, SGD
cfg.OPTIMIZER.WEIGHT_DECAY = 0.0005
cfg.OPTIMIZER.MOMENTUM = 0.9
cfg.OPTIMIZER.LOSS_LR = 0.5

cfg.WARMUP = CfgNode()
cfg.WARMUP.IF_WARMUP = True
cfg.WARMUP.FACTOR = 0.01
cfg.WARMUP.MAX_EPOCHS = 10
cfg.WARMUP.METHOD = 'linear'
cfg.WARMUP.STEPS = (40, 70)
cfg.WARMUP.GAMMA = 0.1

cfg.TRAIN = CfgNode()
cfg.TRAIN.BATCH_SIZE = 64
cfg.TRAIN.LOG_ITER_PERIOD = 100
cfg.TRAIN.MAX_EPOCHS = 150

cfg.EVAL = CfgNode()
cfg.EVAL.EPOCH_PERIOD = 30

cfg.SAVER = CfgNode()
cfg.SAVER.CHECKPOINT_PERIOD = 1
cfg.SAVER.N_SAVED = 1

cfg.TENSORBOARDX = CfgNode()
cfg.TENSORBOARDX.IF_ON = False
cfg.TENSORBOARDX.SCALAR = False
cfg.TENSORBOARDX.HIST = False

cfg.TEST = CfgNode()
cfg.TEST.IF_ON = False
cfg.TEST.BATCH_SIZE = 128  # from evaluate and extract feature
cfg.TEST.IF_RE_RANKING = True
cfg.TEST.IF_FEAT_NORM = True
cfg.TEST.RUN_ID = '02'
cfg.TEST.IF_CLASSIFT_FEATURE = True

cfg.JOINT = CfgNode()
cfg.JOINT.IF_ON = False
cfg.JOINT.DATASET_NAME = ("dukemtmc", "msmt17")

cfg.UDA = CfgNode()
cfg.UDA.IF_ON = False
cfg.UDA.DATASET_NAME = 'dukemtmc'
cfg.UDA.IF_FLIP = True
cfg.UDA.IF_RE_RANKING = True
cfg.UDA.TIMES = 1

cfg.FEAT = CfgNode()
cfg.FEAT.IF_ON = False
cfg.FEAT.DATASET_NAME = 'dukemtmc'

cfg.CONTINUATION = CfgNode()
cfg.CONTINUATION.IF_ON = False
cfg.CONTINUATION.DATASET_NAME = 'dukemtmc'
cfg.CONTINUATION.T = 10.0
cfg.CONTINUATION.LOSS_TYPE = 'ae_dict tr_dist'  # ce_dict tr_dist

cfg.EBLL = CfgNode()
cfg.EBLL.IF_ON = False
cfg.EBLL.IN_PLANES = 2048
cfg.EBLL.CODE_SIZE = 512
cfg.EBLL.OPTIMIZER_BASE_LR = 0.1
cfg.EBLL.LOSS_TYPE = "ae_loss"
cfg.EBLL.LAMBDA = 0.01
cfg.EBLL.DIST_TYPE = "mse"
cfg.EBLL.DATASET_NAME = 'dukemtmc'
cfg.EBLL.MAX_EPOCHS = 50
cfg.EBLL.AE_LOSS_WEIGHT = 0.01
