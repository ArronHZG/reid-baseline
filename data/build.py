"""
@author:  arron
"""
import logging
from collections import OrderedDict

import torch
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


def _get_train_loader(cfg, batch_size, train_set, sampler, shuffle, is_train=True):
    train_transforms = build_transforms(cfg, is_train=is_train)
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
        dataset = init_dataset(name, root=cfg.DATASET.ROOT_DIR, verbose=verbose)
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
    dataset = init_dataset(dataset_name, root=cfg.DATASET.ROOT_DIR)
    batch_size, sampler, shuffle = _get_train_sampler(cfg, dataset.train)
    # print(sampler)
    # i = 0
    # for item in sampler:
    #     print(item)
    #     i += 1
    #     if i == 30:
    #         break
    train_loader = _get_train_loader(cfg, batch_size, dataset.train, sampler, shuffle)
    return train_loader, dataset.num_train_pids


def make_train_data_loader_for_extract(cfg, dataset_name, is_train=False):
    # ######
    # train
    # ######
    dataset = init_dataset(dataset_name, root=cfg.DATASET.ROOT_DIR)
    batch_size, sampler, shuffle = _get_train_sampler(cfg, dataset.train, extract=True)
    train_loader = _get_train_loader(cfg, batch_size, dataset.train, sampler, shuffle, is_train)
    return train_loader, dataset.num_train_pids


def make_train_data_loader_with_labels(cfg, dataset_name, labels):
    dataset = init_dataset(dataset_name, root=cfg.DATASET.ROOT_DIR, verbose=False)
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
        datasets.append(init_dataset(name, root=cfg.DATASET.ROOT_DIR))

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

    logger.info("  ----------------------------------------")
    logger.info("  subset   | # ids | # images | # cameras")
    logger.info("  ----------------------------------------")
    logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_classes, num_train_imgs, num_train_cams))

    batch_size, sampler, shuffle = _get_train_sampler(cfg, all_dataset_train)
    train_loader = _get_train_loader(cfg, batch_size, all_dataset_train, sampler, shuffle)
    return train_loader, num_classes


def _get_target_data_loader(cfg, target_name):
    dataset = init_dataset(target_name, root=cfg.DATASET.ROOT_DIR, verbose=False)
    batch_size, sampler, shuffle = _get_train_sampler(cfg, dataset.train)

    train_transforms = build_transforms(cfg, is_train=True)
    train_set = ImageDataset(dataset.train, train_transforms)

    def train_collate_fn_add_feat(batch):
        imgs, pids, _, _, = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        imgs = torch.stack(imgs, dim=0)
        return imgs, pids

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=train_collate_fn_add_feat)
    return train_loader, dataset.num_train_pids


def _get_feat_data_loader(cfg, source_name, feat):
    dataset = init_dataset(source_name, root=cfg.DATASET.ROOT_DIR, verbose=False)
    generate_train = []
    for i in range(feat.size(0)):
        img_path, _, _ = dataset.train[i]
        generate_train.append((img_path, feat[i], -1))
    dataset.train = generate_train
    dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery)
    batch_size = cfg.TRAIN.BATCH_SIZE

    train_transforms = build_transforms(cfg, is_train=False)
    train_set = ImageDataset(dataset.train, train_transforms)

    def train_collate_fn_by_feat(batch):
        imgs, feats, _, _, = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        feats = torch.stack(feats, dim=0)
        return imgs, feats

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=train_collate_fn_by_feat)
    return train_loader, dataset.num_train_pids


class DataLoaderWithFeat:
    def __init__(self, target_dataloader, source_dataloader, batch_size):
        self.target_dataloader = target_dataloader
        self.source_dataloader = source_dataloader
        self.target = iter(target_dataloader)
        self.source = iter(source_dataloader)
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        try:
            t_imgs, t_pids = next(self.target)
        except StopIteration:
            self.target = iter(self.target_dataloader)
            t_imgs, t_pids = next(self.target)

        try:
            s_imgs, s_feats = next(self.source)
        except StopIteration:
            self.source = iter(self.source_dataloader)
            s_imgs, s_feats = next(self.source)

        return t_imgs, t_pids, s_imgs, s_feats
        # return torch.cat([t_imgs, s_imgs]), \
        #        torch.cat([t_pids, s_pids]), \
        #        torch.cat([t_feats, s_feats])

    def __len__(self):
        return len(self.target_dataloader)


def make_data_with_loader_with_feat_label(cfg, source_name, target_name, feat):
    source_dataloader, _ = _get_feat_data_loader(cfg, source_name, feat)
    target_dataloader, target_num = _get_target_data_loader(cfg, target_name)
    dataloader = DataLoaderWithFeat(target_dataloader, source_dataloader, cfg.TRAIN.BATCH_SIZE)
    return dataloader, target_num


if __name__ == '__main__':
    from config import cfg

    source_name = "market1501"
    feat = torch.zeros(12936, 2048)
    a, b = make_train_data_loader(cfg, source_name)
    # make_data_with_loader_with_feat_label(cfg, source_name, target_name, feat)

# class MyNumbers:
#     def __init__(self, max):
#         self.max = max
#
#     def __iter__(self):
#         self.a = 1
#         return self
#
#     def __next__(self):
#         if self.a <= self.max:
#             x = self.a
#             self.a += 1
#             return x
#         else:
#             raise StopIteration
#
#
# myclass = MyNumbers(20)
# myiter = iter(myclass)
#
# myclass1 = MyNumbers(5)
# myiter1 = iter(myclass1)
#
# while (1):
#     print(next(myiter))
#     try:
#         print(next(myiter1))
#     except StopIteration:
#         myiter1 = iter(myclass1)
#         print(next(myiter1))
