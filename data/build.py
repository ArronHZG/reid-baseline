# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms


def make_data_loader(cfg, cluster=False):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    # TODO: add multi dataset to train
    if not cluster:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        dataset = init_dataset(cfg.CLUSTER.DATASETS_NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    shuffle = True
    sampler = None
    if cfg.DATALOADER.SAMPLER is 'RandomIdentity':
        shuffle = False
        sampler = RandomIdentitySampler(dataset.train, cfg.TRAIN.BATCH_SIZE, cfg.DATALOADER.NUM_INSTANCE)

    if cfg.LOSS.LOSS_TYPE is not 'softmax' and cfg.DATALOADER.SAMPLER is 'None':
        raise ValueError(f"Loss {cfg.LOSS.LOSS_TYPE} should not using {cfg.DATALOADER.SAMPLER} dataloader sampler")

    batch_size = cfg.TRAIN.BATCH_SIZE
    if cfg.CLUSTER.IF_ON:
        batch_size = cfg.TEST.BATCH_SIZE

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=train_collate_fn)

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn)
    return train_loader, val_loader, len(dataset.query), num_classes
