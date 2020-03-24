#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/17 15:00
# @Author  : Hao Luo
# @File    : msmt17.py

import glob
import logging
import re

import os.path as osp

from .bases import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html
    download: https://pan.baidu.com/s/1-kYPIGib7FFnk8qY49Yzuw

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'msmt17'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(MSMT17, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # self.train_dir = osp.join(self.dataset_dir, 'MSMT17_V2/mask_train_v2')
        # self.test_dir = osp.join(self.dataset_dir, 'MSMT17_V2/mask_test_v2')
        # self.list_train_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_train.txt')
        # self.list_val_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_val.txt')
        # self.list_query_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_query.txt')
        # self.list_gallery_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_gallery.txt')
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        # train = self._process_dir(self.train_dir, self.list_train_path)
        # # val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path)
        # query = self._process_dir(self.test_dir, self.list_query_path)
        # gallery = self._process_dir(self.test_dir, self.list_gallery_path)

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            logger = logging.getLogger("reid_baseline.dataset")
            logger.info("MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    # def _check_before_run(self):
    #     """Check if all files are available before going deeper"""
    #     if not osp.exists(self.dataset_dir):
    #         raise RuntimeError("'{}' is not available".format(self.dataset_dir))
    #     if not osp.exists(self.train_dir):
    #         raise RuntimeError("'{}' is not available".format(self.train_dir))
    #     if not osp.exists(self.test_dir):
    #         raise RuntimeError("'{}' is not available".format(self.test_dir))

    # def _process_dir(self, dir_path, list_path):
    #     with open(list_path, 'r') as txt:
    #         lines = txt.readlines()
    #     dataset = []
    #     pid_container = set()
    #     for img_idx, img_info in enumerate(lines):
    #         img_path, pid = img_info.split(' ')
    #         pid = int(pid)  # no need to relabel
    #         camid = int(img_path.split('_')[2])
    #         img_path = osp.join(dir_path, img_path)
    #         dataset.append((img_path, pid, camid))
    #         pid_container.add(pid)
    #
    #     # check if pid starts from 0 and increments with 1
    #     for idx, pid in enumerate(pid_container):
    #         assert idx == pid, "See code comment for explanation"
    #     return dataset

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
