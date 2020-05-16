import logging

import torch
from ignite.engine import Engine, Events

from tools.component import TrainComponent
from utils.tensor_utils import batch_horizontal_flip

logger = logging.getLogger("reid_baseline.extract")


def create_extractor(model, device, flip=False):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            img, target = batch
            img = img.to(device)
            target = target.to(device)
            feat_t, feat_c = model(img)
            if flip:
                flip_img = batch_horizontal_flip(img, device)
                flip_feat = model(flip_img)
                feat_c += flip_feat

            return feat_t, target

    engine = Engine(_inference)
    return engine


def do_extract(cfg,
               data_loader,
               tr_comp: TrainComponent,
               flip=False):
    device = cfg.MODEL.DEVICE

    features = []
    labels = []

    extractor = create_extractor(tr_comp.model, device, flip)

    @extractor.on(Events.ITERATION_COMPLETED)
    def get_output(engine):
        feat, label = extractor.state.output
        # feat = torch.nn.functional.normalize(feat, dim=1, p=2)
        features.append(feat)
        labels.append(label)

    extractor.run(data_loader)

    cat_features = torch.cat(features)
    cat_labels = torch.cat(labels)

    return cat_features, cat_labels
