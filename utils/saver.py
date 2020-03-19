import glob
import os
import numpy as np
import torch
from ignite.handlers import ModelCheckpoint


class Saver:

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.SAVER.OUTPUT_DIR is not "":
            self.save_path = cfg.SAVER.OUTPUT_DIR
        else:
            self.save_path = os.path.join("..",
                                          "run",
                                          cfg.DATASETS.NAMES,
                                          cfg.MODEL.NAME)

        self.runs = glob.glob(os.path.join(self.save_path, 'experiment-*'))
        run_ids = sorted([int(experiment.split('-')[-1]) for experiment in self.runs]) if self.runs else [0]
        run_id = run_ids[-1] + 1
        self.save_path = os.path.join(self.save_path, 'experiment-{}'.format(str(run_id)))

        self.train_checkpointer = ModelCheckpoint(self.save_path,
                                                  "train",
                                                  n_saved=cfg.SAVER.N_SAVED,
                                                  require_empty=False)

        self.best_checkpointer = ModelCheckpoint(self.save_path, "best", require_empty=False)

        self.to_save = None

        self.best_result = 0

    def load_checkpoint(self, is_best=False):
        try:
            if is_best:
                checkpoint = torch.load(self.fetch_checkpoint_model_filename("train"))
            else:
                checkpoint = torch.load(self.fetch_checkpoint_model_filename("best"))
        except Exception:
            raise RuntimeError("checkpoint doesn't exist.")

        self.train_checkpointer.load_objects(self.to_save, checkpoint)
        self.to_save['trainer'].state.max_epochs = self.cfg.TRAIN.MAX_EPOCHS
        self.best_result = np.load(os.path.join(self.save_path, "best.npy"))

    # loading the saved model
    # TODO 增加run_id
    def fetch_checkpoint_model_filename(self, prefix):
        checkpoint_files = os.listdir(self.save_path)
        checkpoint_files = [f for f in checkpoint_files if '.pth' in f and prefix in f]
        checkpoint_iter = [int(x.split('_')[2].split('.')[0]) for x in checkpoint_files]
        last_idx = np.array(checkpoint_iter).argmax()
        return os.path.join(self.save_path, checkpoint_files[last_idx])

    def save_best_value(self, value):
        self.best_result = value
        best_name_list = glob.glob(os.path.join(self.save_path, "*.npy"))
        if len(best_name_list) > 0:
            os.remove(best_name_list[0])
        np.save(os.path.join(self.save_path, f"best-{value:.4f}.npy"), self.best_result)
