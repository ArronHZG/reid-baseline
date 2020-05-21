"""
@author:  arron
@contact: hou.arron@gmail.com


    module
    data
for:
    eval
    extract
    calculate dist
    cluster label
    generate data
    fine tuning
"""
import logging

import torch
from torch import nn
from ignite.engine import Engine, Events
from sklearn.cluster import DBSCAN

from engine.extract import extract_features
from engine.inference import inference
from engine.trainer import do_train
from modeling import build_model
from utils.tensor_utils import euclidean_dist, batch_horizontal_flip
from utils.re_ranking import re_ranking

logger = logging.getLogger("reid_baseline.cluster")


def compute_dist(feat, if_re_ranking):
    if if_re_ranking:
        dist_matrix = re_ranking(feat, feat, k1=20, k2=6, lambda_value=0.3)
        dist_matrix = torch.from_numpy(dist_matrix)
    else:
        dist_matrix = euclidean_dist(feat, feat)
    dist_matrix.clamp_(min=1e-12, max=1e+12)
    return dist_matrix


cluster = None


def create_cluster(dist: torch.Tensor):
    """
    If want to use DBSCAN, we need to affirm the best epsilon in DBSCAN.
    :param dist:
    :return:
    """
    rho = 1.6e-3
    dist = dist.triu(1)  # the upper triangular part of a matrix
    dist = dist.view(dist.size(0) ** 2, -1).squeeze()
    dist = dist[dist.nonzero()].squeeze()  # get all distance  dim = 1
    sorted_dist, _ = dist.sort()
    top_num = torch.tensor(rho * sorted_dist.size()[0]).round().to(torch.int)
    if top_num <= 20:
        top_num = 20
    logger.info(f"top_num: {top_num}")
    eps = sorted_dist[:top_num].mean().cpu().numpy()
    # logger.info(f'eps in cluster: {eps:.3f}')
    logger.info(f"eps: {eps:.4f}")
    return DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)


def generate_self_label(dist_matrix):
    global cluster
    if cluster is None:
        logger.info("Create cluster function")
        cluster = create_cluster(dist_matrix)

    dist = dist_matrix.cpu().numpy()

    labels = cluster.fit_predict(dist)
    return labels


def load_weight(old_model: nn.Module, new_model: nn.Module):
    old_weight = old_model.state_dict()
    old_weight.pop("classifier.weight")
    new_model.load_state_dict(old_weight, strict=False)


class SSG:
    def __init__(self, cfg, saver):
        self.cfg = cfg
        self.saver = saver
        self.device = self.cfg.MODEL.DEVICE

        # define module and load module weight, this module only is used to extract feature.
        logger.info(f"load module from {self.saver.load_dir}")
        self.model = build_model(self.cfg, 0)
        if self.cfg.MODEL.DEVICE is 'cuda':
            self.model.cuda()
        self.saver.to_save = {'module': self.model}
        self.saver.load_checkpoint(is_best=True)
        saver.best_result = 0

        # data
        self.target_train_loader, self.val_loader, self.num_query, _ = make_data_loader(cfg, cluster=True)

    def cluster(self):
        # eval
        inference(self.cfg, self.model, self.val_loader, self.num_query)
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # extract feature from target dataset
        logger.info("Extract feature")
        target_features, _ = extract_features(self.model, self.device, self.target_train_loader, self.cfg.UDA.IF_FLIP)
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        logger.info("Compute dist")
        dict_matrix = compute_dist(target_features, if_re_ranking=self.cfg.UDA.IF_RE_RANKING)
        # dict_matrix = compute_dist(target_features, if_re_ranking=False)
        del target_features

        # generate label
        logger.info("Cluster self label")
        labels = generate_self_label(dict_matrix)

        # generate_dataloader
        logger.info("Generate data loader")
        gen_train_loader, _, _, gen_num_classes = make_data_loader(self.cfg, labels=labels)
        logger.info(f"class num {gen_num_classes}")

        # train
        self.cluster_train(gen_train_loader, gen_num_classes)

    def cluster_train(self, train_loader, num_classes):
        tr_comp = TrainComponent(self.cfg, num_classes)
        load_weight(self.model, tr_comp.model)
        del self.model
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        do_train(self.cfg,
                 tr_comp.model,
                 train_loader,
                 self.val_loader,
                 tr_comp.optimizer,
                 tr_comp.scheduler,
                 tr_comp.loss_function,
                 self.num_query,
                 self.saver,
                 center_criterion=tr_comp.loss_center,
                 optimizer_center=tr_comp.optimizer_center)
        self.model = tr_comp.model


def do_uda(cfg, saver):
    ssg = SSG(cfg, saver)
    for i in range(cfg.UDA.TIMES):
        logger.info(f"############# times {i} ###############")
        ssg.cluster()
