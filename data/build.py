"""
@author:  arron
"""
import logging
from collections import OrderedDict

from torch.utils.data import DataLoader

from data.collate_batch import train_collate_fn, val_collate_fn
from data.datasets import init_dataset, ImageDataset
from data.samplers import RandomIdentitySampler
from data.transforms import build_transforms

logger = logging.getLogger("reid_baseline.dataset")


def _get_train_sampler(cfg, train_set, extract=False):
    shuffle = True
    sampler = None
    if not extract:
        if cfg.DATALOADER.SAMPLER is 'RandomIdentity':
            shuffle = False
            sampler = RandomIdentitySampler(train_set, cfg.TRAIN.BATCH_SIZE, cfg.DATALOADER.NUM_INSTANCE)
        if cfg.LOSS.LOSS_TYPE is not 'softmax' and cfg.DATALOADER.SAMPLER is 'None':
            raise ValueError(f"Loss {cfg.LOSS.LOSS_TYPE} should not using {cfg.DATALOADER.SAMPLER} dataloader sampler")
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        shuffle = False
        sampler = None
        batch_size = cfg.TEST.BATCH_SIZE

    return batch_size, sampler, shuffle


def _get_train_loader(cfg, batch_size, train_set, sampler, shuffle):
    train_transforms = build_transforms(cfg, is_train=True)
    train_set = ImageDataset(train_set, train_transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=train_collate_fn)
    return train_loader


def make_multi_valid_data_loader(cfg, data_set_names, verbose=False):
    valid = OrderedDict()
    for name in data_set_names:
        dataset = init_dataset(name, root=cfg.DATASETS.ROOT_DIR, verbose=verbose)
        val_transforms = build_transforms(cfg, is_train=False)
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=val_collate_fn)
        valid[name] = (val_loader, len(dataset.query))

    return valid


def make_train_data_loader(cfg, dataset_name):
    # ######
    # train
    # ######
    dataset = init_dataset(dataset_name, root=cfg.DATASETS.ROOT_DIR)
    batch_size, sampler, shuffle = _get_train_sampler(cfg, dataset.train)
    train_loader = _get_train_loader(cfg, batch_size, dataset.train, sampler, shuffle)
    return train_loader, dataset.num_train_pids


def make_train_data_loader_for_extract(cfg, dataset_name):
    # ######
    # train
    # ######
    dataset = init_dataset(dataset_name, root=cfg.DATASETS.ROOT_DIR)
    batch_size, sampler, shuffle = _get_train_sampler(cfg, dataset.train, extract=True)
    train_loader = _get_train_loader(cfg, batch_size, dataset.train, sampler, shuffle)
    return train_loader, dataset.num_train_pids


def make_train_data_loader_with_labels(cfg, dataset_name, labels):
    dataset = init_dataset(dataset_name, root=cfg.DATASETS.ROOT_DIR, verbose=False)
    generate_train = []
    for i in range(len(labels)):
        if labels[i] == -1:
            continue
        img_path, _, _ = dataset.train[i]
        generate_train.append((img_path, labels[i], -1))
    dataset.train = generate_train
    dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery)

    batch_size, sampler, shuffle = _get_train_sampler(cfg, dataset.train)
    train_loader = _get_train_loader(cfg, batch_size, dataset.train, sampler, shuffle)
    return train_loader, dataset.num_train_pids


def make_train_data_loader_with_expand(cfg, data_set_names):
    # ######
    # train
    # ######
    datasets = []
    for name in data_set_names:
        datasets.append(init_dataset(name, root=cfg.DATASETS.ROOT_DIR))

    all_dataset_train = []
    num_classes = 0
    num_train_imgs = 0
    num_train_cams = 0

    for dataset in datasets:
        for sample in dataset.train:
            temp = (sample[0], sample[1] + num_classes, sample[2] + num_train_cams)
            all_dataset_train.append(temp)
        num_classes += dataset.num_train_pids
        num_train_imgs += dataset.num_train_imgs
        num_train_cams += dataset.num_train_cams

    logger.info("Dataset statistics:")
    logger.info("  ----------------------------------------")
    logger.info("  subset   | # ids | # images | # cameras")
    logger.info("  ----------------------------------------")
    logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_classes, num_train_imgs, num_train_cams))

    batch_size, sampler, shuffle = _get_train_sampler(cfg, all_dataset_train)
    train_loader = _get_train_loader(cfg, batch_size, all_dataset_train, sampler, shuffle)
    return train_loader, num_classes


def make_data_with_loader_with_feat(cfg, dataset_name, feat):
    dataset = init_dataset(dataset_name, root=cfg.DATASETS.ROOT_DIR, verbose=False)
    generate_train = []
    for i in range(len(feat)):
        if feat[i] == -1:
            continue
        img_path, _, _ = dataset.train[i]
        generate_train.append((img_path, feat[i], -1))
    dataset.train = generate_train
    dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery)

    batch_size, sampler, shuffle = _get_train_sampler(cfg, dataset.train)
    train_loader = _get_train_loader(cfg, batch_size, dataset.train, sampler, shuffle)
    return train_loader, dataset.num_train_pids
