import os
import shutil

from modeling.strong_baseline import model_map
from utils.logger import setup_logger
from test.model_shape import count_param, show_model


def test_backbone(root_dir, model_name, se=False, ibn_a=False, ibn_b=False):
    base = model_map[model_name](last_stride=1, pretrained=False, se=se, ibn_a=ibn_a, ibn_b=ibn_b)
    if se:
        model_name += "_se"

    if ibn_a:
        model_name += "_ibn_a"

    if ibn_b:
        model_name += "_ibn_b"

    print(f"{'*' * 70}\n{'*' * 70}\n{model_name}")

    saver_path = os.path.join(root_dir, model_name)

    if not os.path.exists(saver_path):
        os.makedirs(saver_path)
    else:
        shutil.rmtree(saver_path)
        os.makedirs(saver_path)

    show_model(base, saver_path, input_size=(3, 256, 128))


if __name__ == '__main__':
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }
    names = model_urls.keys()

    bool_list = [[True, False, False], [True, True, False], [True, False, True], [False, True, False],
                 [False, False, True]]

    root_dir = os.path.join("..", "..", "delving_into_model")

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    setup_logger("module", root_dir, 0)

    for name in names:
        for b in bool_list:
            test_backbone(root_dir, name, se=b[0], ibn_a=b[1], ibn_b=b[2])
