# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
from ignite.metrics import Metric

from evaluation.rank import eval_func

logger = logging.getLogger("reid_baseline.R1_mAP")


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, if_feat_norm=True, if_re_rank=False):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.if_feat_norm = if_feat_norm
        self.if_re_rank = if_re_rank
        self.logger = logging.getLogger("reid_baseline.R1_mAP")

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(pid)
        self.camids.extend(camid)

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        pids = self.pids
        camids = self.camids

        if self.if_feat_norm:
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = pids[:self.num_query]
        q_camids = camids[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        g_pids = pids[self.num_query:]
        g_camids = self.camids[self.num_query:]

        # distmat = euclidean_dist(qf, gf).cpu().numpy()
        # distmat = re_ranking(qf, gf, k1=24, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(qf, gf, q_pids, g_pids, q_camids, g_camids, distance_type='euclidean', re_rank=False)
        return cmc, mAP

