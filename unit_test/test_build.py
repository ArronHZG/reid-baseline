from unittest import TestCase

import torch

from config import cfg

from data.build import generate_data_loader


class Test(TestCase):
    def test_generate_data_loader(self):
        generate_data_loader(cfg, 5,
                             torch.randint(1, 100, [1, 10000]))
