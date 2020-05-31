from modeling.EBLL import EBLL
from modeling.backbone.autoencoder import AutoEncoder
from modeling.strong_baseline import Baseline


def build_model(cfg, num_classes):
    args = {'num_classes': num_classes,
            'last_stride': cfg.MODEL.LAST_STRIDE,
            'model_name': cfg.MODEL.NAME,
            'pretrain_choice': cfg.MODEL.PRETRAIN_CHOICE,
            'se': cfg.MODEL.IF_SE,
            'ibn_a': cfg.MODEL.IF_IBN_A,
            'ibn_b': cfg.MODEL.IF_IBN_B}
    model = Baseline(**args)
    return model
