from modeling.EBLL import EBLL
from modeling.strong_baseline import Baseline


def build_model(cfg, num_classes):
    args = {'num_classes': num_classes,
            'last_stride': cfg.MODEL.LAST_STRIDE,
            'model_name': cfg.MODEL.NAME,
            'pretrain_choice': cfg.MODEL.PRETRAIN_CHOICE,
            'se': cfg.MODEL.IF_SE,
            'ibn_a': cfg.MODEL.IF_IBN_A,
            'ibn_b': cfg.MODEL.IF_IBN_B}
    if cfg.EBLL.IF_ON:
        args['code_size'] = cfg.EBLL.CODE_SIZE
        model = EBLL(**args)
    else:
        model = Baseline(**args)
    return model


def fun(a, b, c=3):
    return a + b + c


if __name__ == '__main__':
    a = {'a': 1, 'b': 1, 'c': 1}
    d = fun(**a)
    print(d)
