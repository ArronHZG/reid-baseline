import glob
import os
from collections import Mapping

import numpy as np
import torch
from ignite.handlers import ModelCheckpoint


class Saver:

    def __init__(self, cfg):
        """
        All Saver based on two dir: save_dir and load_dir.
        for train : save_dir = load_dir and every time This will make a new one.
        for test : save_dir = None , load_dir will automatic be stitched with run_id
        for uda : every time save_dir will be made. and load_dir will automatic be stitched with run_id and source dataset
        """
        self.cfg = cfg
        self.save_dir = ''
        self.load_dir = ''

        dirname_list = os.path.dirname(__file__).split("/")[:-1]
        self.up_dir = "/".join(dirname_list)

        source_name, source_mid_name, target_name, target_mid_name = self._get_some_dir_name(cfg)

        if cfg.TEST.IF_ON:
            self.load_dir = self.get_load_dir(source_mid_name, source_name)
            self.save_dir = self.load_dir
        else:
            self.save_dir = self.get_save_dir(target_mid_name, target_name)
            if target_mid_name == 'direct':
                self.load_dir = self.save_dir
            else:
                self.load_dir = self.get_load_dir(source_mid_name, source_name)

        print(f"save dir: {self.save_dir}")
        print(f"load dir: {self.load_dir}")

        self.train_checkpointer = ModelCheckpoint(self.save_dir,
                                                  "train",
                                                  n_saved=cfg.SAVER.N_SAVED,
                                                  require_empty=False)

        self.best_checkpointer = ModelCheckpoint(self.save_dir,
                                                 "best",
                                                 require_empty=False)

        self.to_save = None

        self.best_result = 0

    def _get_some_dir_name(self, cfg):

        source_name = cfg.DATASET.NAME

        if cfg.CONTINUATION.IF_ON:
            target_name = cfg.CONTINUATION.DATASET_NAME
            target_mid_name = 'continuation'

        elif cfg.FEAT.IF_ON:
            target_name = cfg.FEAT.DATASET_NAME
            target_mid_name = 'feat'

        elif cfg.UDA.IF_ON:
            target_name = cfg.UDA.DATASET_NAME
            target_mid_name = 'uda'

        elif cfg.EXPAND.IF_ON:
            name = cfg.DATASET.NAME
            for n in cfg.EXPAND.DATASET_NAME:
                name += f"--{n}"
            target_name = name
            target_mid_name = 'expand'

        else:
            target_name = cfg.DATASET.NAME
            target_mid_name = 'direct'

        source_mid_name = "direct"

        return source_name, source_mid_name, target_name, target_mid_name

    def get_load_dir(self, mid_name, name):
        load_dir = os.path.join(self.up_dir,
                                "run",
                                mid_name,
                                name,
                                self.cfg.MODEL.NAME)
        run_id = self.cfg.TEST.RUN_ID
        print(f"Loading run_id: {run_id}")
        load_dir = os.path.join(load_dir, f'experiment-{run_id}')
        assert os.path.exists(load_dir), load_dir
        return load_dir

    def get_save_dir(self, mid_name, name):
        first_dir = os.path.join(self.up_dir,
                                 "run",
                                 mid_name,
                                 name,
                                 self.cfg.MODEL.NAME)
        runs = glob.glob(os.path.join(first_dir, 'experiment-*'))
        run_ids = sorted([int(experiment.split('-')[-1]) for experiment in runs]) if runs else [0]
        run_id = run_ids[-1] + 1
        return os.path.join(first_dir, 'experiment-{}'.format(str(run_id).zfill(2)))

    def load_checkpoint(self, is_best=False):
        try:
            if is_best:
                checkpoint = torch.load(self.fetch_checkpoint_model_filename("best"))
                checkpoint['model'].pop("classifier.weight")
            else:
                checkpoint = torch.load(self.fetch_checkpoint_model_filename("train"))
        except Exception:
            raise RuntimeError("checkpoint doesn't exist.")

        self.load_objects(self.to_save, checkpoint)
        # self.best_result = np.load(glob.glob(os.path.join(self.load_dir, "*.npy"))[0])
        if 'trainer' in self.to_save.keys():
            self.to_save['trainer'].state.max_epochs = self.cfg.TRAIN.MAX_EPOCHS

    # loading the saved model
    def fetch_checkpoint_model_filename(self, prefix):
        checkpoint_files = os.listdir(self.load_dir)
        checkpoint_files = [f for f in checkpoint_files if '.pth' in f and prefix in f]
        checkpoint_iter = [int(x.split('_')[2].split('.')[0]) for x in checkpoint_files]
        last_idx = np.array(checkpoint_iter).argmax()
        return os.path.join(self.load_dir, checkpoint_files[last_idx])

    def save_best_value(self, value):
        self.best_result = value
        best_name_list = glob.glob(os.path.join(self.save_dir, "*.npy"))
        if len(best_name_list) > 0:
            os.remove(best_name_list[0])
        np.save(os.path.join(self.save_dir, f"best-{value:.4f}.npy"), self.best_result)

    @staticmethod
    def load_objects(to_load: Mapping, checkpoint: Mapping) -> None:
        """Helper method to apply `load_state_dict` on the objects from `to_load` using states from `checkpoint`.

        Args:
            to_load (Mapping): a dictionary with objects, e.g. `{"model": model, "optimizer": optimizer, ...}`
            checkpoint (Mapping): a dictionary with state_dicts to load, e.g. `{"model": model_state_dict,
                "optimizer": opt_state_dict}`. If `to_load` contains a single key, then checkpoint can contain directly
                corresponding state_dict.
        """
        if len(to_load) == 1:
            # single object and checkpoint is directly a state_dict
            key, obj = list(to_load.items())[0]
            if key not in checkpoint:
                obj.load_state_dict(checkpoint)
                return

        # multiple objects to load
        for k, obj in to_load.items():
            if k not in checkpoint:
                raise ValueError("Object labeled by '{}' from `to_load` is not found in the checkpoint".format(k))
            obj.load_state_dict(checkpoint[k], strict=False)


if __name__ == '__main__':
    from config import cfg

    Saver(cfg)
