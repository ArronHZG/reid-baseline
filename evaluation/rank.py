# credits: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/rank.py

import warnings

from .rank_py import evaluate_py

try:
    from .rank_cylib.rank_cy import evaluate_cy

    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython rank evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def eval_func(*args, **kwargs):
    if IS_CYTHON_AVAI and False:
        return evaluate_cy(*args, **kwargs)
    else:
        return evaluate_py(*args, **kwargs)
