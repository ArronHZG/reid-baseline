from unittest import TestCase

import torch

from evaluation.rank import eval_func


class TestR1_mAP(TestCase):
    def test_R1_mAP(self):
        num_q = 30
        num_g = 300
        dim = 512
        max_rank = 5
        g_feats = torch.randn((num_g, dim)) * 20
        g_feats = g_feats / torch.nn.functional.normalize(g_feats)
        q_feats = torch.randn((num_q, dim)) * 20
        q_feats = q_feats / torch.nn.functional.normalize(q_feats)
        q_pids = torch.randint(0, num_q, size=[num_q])
        g_pids = torch.randint(0, num_g, size=[num_g])
        q_camids = torch.randint(0, 5, size=[num_q])
        g_camids = torch.randint(0, 5, size=[num_g])

        result = eval_func(q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank=max_rank,
                                 distance_type='DSR', re_rank=False)
        print(result)
