# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def make_data_loader(cfg, cluster=False, labels=None):
    # ######
    # train
    # ######
    if labels is None:
        labels = []
    labels_flag = len(labels) > 0
    train_transforms = build_transforms(cfg, is_train=True)
    # TODO: add multi dataset to train
    dataset = None

    # for simple train
    if not cluster and not labels_flag:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    # get uda data set for extracting features
    if cluster and not labels_flag:
        dataset = init_dataset(cfg.UDA.DATASETS_NAMES, root=cfg.DATASETS.ROOT_DIR)

    # using generate labels for uda train
    if not cluster and labels_flag:
        dataset = init_dataset(cfg.UDA.DATASETS_NAMES, root=cfg.DATASETS.ROOT_DIR, verbose=False)
        generate_train = []
        for i in range(len(labels)):
            if labels[i] == -1:
                continue
            img_path, _, _ = dataset.train[i]
            generate_train.append((img_path, labels[i], -1))
        dataset.train = generate_train
        dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery)

    if cluster and labels_flag:
        raise ValueError(f" not support")

    num_classes = dataset.num_train_pids if not labels_flag else len(set(labels)) - 1
    train_set = ImageDataset(dataset.train, train_transforms)
    shuffle = True
    sampler = None
    if not cluster:
        if cfg.DATALOADER.SAMPLER is 'RandomIdentity':
            shuffle = False
            sampler = RandomIdentitySampler(dataset.train, cfg.TRAIN.BATCH_SIZE, cfg.DATALOADER.NUM_INSTANCE)
        if cfg.LOSS.LOSS_TYPE is not 'softmax' and cfg.DATALOADER.SAMPLER is 'None':
            raise ValueError(f"Loss {cfg.LOSS.LOSS_TYPE} should not using {cfg.DATALOADER.SAMPLER} dataloader sampler")
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        shuffle = False
        sampler = None
        batch_size = cfg.TEST.BATCH_SIZE

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=train_collate_fn)
    # ######
    # valid
    # ######
    val_transforms = build_transforms(cfg, is_train=False)
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn)
    query_num = len(dataset.query)
    return train_loader, val_loader, query_num, num_classes
