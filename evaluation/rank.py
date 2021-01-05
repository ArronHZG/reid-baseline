# credits: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/rank.py

import warnings

import torch
import numpy as np

from .rank_py import evaluate_py, DISTANCE_TYPES

try:
    from .rank_cylib.rank_cy import evaluate_cy

    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython rank evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def call_cy(qf, gf,
            q_pids, g_pids,
            q_camids, g_camids,
            max_rank=50,
            distance_type='euclidean',
            re_rank=False):
    indices = DISTANCE_TYPES[distance_type](qf, gf)

    np.save('qf.npy', np.asarray(q_pids))
    np.save('qf_feat.npy', qf.cpu().numpy())

    return evaluate_cy(indices,
                       qf.cpu().numpy(),
                       gf.cpu().numpy(),
                       q_pids,
                       g_pids,
                       q_camids,
                       g_camids,
                       max_rank)


def eval_func(*args, **kwargs):
    if IS_CYTHON_AVAI:
        return call_cy(*args, **kwargs)
    else:
        return evaluate_py(*args, **kwargs)
