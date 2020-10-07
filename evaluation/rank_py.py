# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import faiss
import numpy as np
import torch

from utils.tensor_utils import euclidean_dist, cosine_dist


def get_euclidean_indices(qf, gf):
    distmat = euclidean_dist(qf, gf).cpu().numpy()
    return np.argsort(distmat, axis=1)


def get_cosine_indices(qf, gf):
    distmat = cosine_dist(qf, gf).cpu().numpy()
    return np.argsort(distmat, axis=1)


def get_DSR_indices(qf, gf):
    qf = qf.cpu().numpy()
    gf = gf.cpu().numpy()

    num_g = gf.shape[0]
    dim = qf.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(gf)
    return index.search(qf, k=num_g)[1]


DISTANCE_TYPES = {
    'euclidean': get_euclidean_indices,
    'cosine': get_cosine_indices,
    'DSR': get_DSR_indices
}


def evaluate_py(qf: torch.Tensor, gf: torch.Tensor,
                q_pids: torch.Tensor, g_pids: torch.Tensor,
                q_camids: torch.Tensor, g_camids: torch.Tensor,
                max_rank=50,
                distance_type='euclidean',
                re_rank=False):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    num_q, num_g = qf.size(0), gf.size(0)
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = DISTANCE_TYPES[distance_type](qf, gf)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
