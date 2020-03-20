import logging

import torch
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from data import make_data_loader
from tools.expand import TrainComponent


def create_extractor(model, device=None, flip=False):
    def horizontal_flip(tensor):
        """
        :param tensor: N x C x H x W
        :return:
        """
        inv_idx = torch.arange(tensor.size(3) - 1, -1, -1).long()
        img_flip = tensor.index_select(3, inv_idx)
        return img_flip

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            img, target = batch
            img = img.to(device)
            target = target.to(device)
            feat = model(img)
            if flip:
                flip_img = horizontal_flip(img)
                flip_feat = model(flip_img)
                feat += flip_feat

            norm = torch.norm(feat, p=2, dim=1, keepdim=True)
            feat = feat.div(norm.expand_as(feat))

            return feat, target

    engine = Engine(_inference)
    return engine


def extract_features(model, device, data_loader, flip):
    features = []
    labels = []

    extractor = create_extractor(model, device, flip)

    @extractor.on(Events.ITERATION_COMPLETED)
    def adjust_learning_rate(engine):
        feat, label = extractor.state.output
        features.append(feat.cpu())
        labels.append(label.cpu())

    pbar = ProgressBar(persist=True)
    pbar.attach(extractor)

    extractor.run(data_loader)

    return features, labels


def compute_dist(source_features, target_features, lambda_value, if_re_ranking):
    features = []
    labels = []
    return features, labels


def generate_selflabel(euclidean_dist_list, rerank_dist_list, iter_n, args, cluster_list):
    features = []
    labels = []
    return features, labels


def generate_dataloader(tgt_dataset, labels_list, train_transformer, iter_n, args):
    return []


def cluster_train():
    pass
    # do_train(cfg,
    #     model,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     scheduler,
    #     loss_fn,
    #     num_query,
    #     saver,
    #     center_criterion=None,
    #     optimizer_center=None)


def ssg(cfg, saver):
    logger = logging.getLogger("reid_baseline.cluster")

    # data
    source_train_loader, source_val_loader, source_num_query, source__num_classes \
        = make_data_loader(cfg)
    target_train_loader, target_val_loader, target_num_query, target_num_classes \
        = make_data_loader(cfg, cluster=True)

    # define model and load model weight
    tr_comp = TrainComponent(cfg,
                             logger,
                             source__num_classes)
    to_load = {'model': tr_comp.model}
    saver.to_save = to_load
    saver.load_checkpoint(is_best=True)

    # extract feature from source and target dataset
    source_features, _ = extract_features(tr_comp.model, tr_comp.device, source_train_loader,
                                          cfg.CLUSTER.IF_FLIP)
    print(source_features[0])
    print(source_features[0].size())
    return
    target_features, _ = extract_features(tr_comp.model, tr_comp.device, target_train_loader,
                                          cfg.CLUSTER.IF_FLIP)

    euclidean_dist_list, re_rank_dist_list = compute_dist(source_features, target_features, lambda_value=1,
                                                          if_re_ranking=True)
    del target_features
    del source_features
    labels_list, cluster_list = generate_selflabel(
        euclidean_dist_list, rerank_dist_list, iter_n, args, cluster_list)

    train_loader_list = generate_dataloader(tgt_dataset, labels_list, train_transformer, iter_n, args)
    del labels_list

    cluster_train()


def do_cluster(cfg, saver):
    for i in range(1):
        ssg(cfg, saver)
