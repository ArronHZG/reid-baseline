import logging

import torch
from ignite.engine import Engine

from engine.trainer import run
from loss import Loss
from utils.component import TrainComponent

logger = logging.getLogger("reid_baseline.continue")


def do_continuous_train(cfg,
                        train_loader,
                        valid_dict,
                        source_tr_comp: TrainComponent,
                        current_tr_comp: TrainComponent,
                        saver):
    # tb_log = TensorBoardXLog(cfg, saver.save_dir)

    trainer = create_supervised_trainer(source_tr_comp.model,
                                        current_tr_comp.model,
                                        current_tr_comp.optimizer,
                                        current_tr_comp.loss,
                                        device=cfg.MODEL.DEVICE,
                                        apex=cfg.APEX.IF_ON)

    run(cfg, train_loader, valid_dict, current_tr_comp, saver, trainer)


def create_supervised_trainer(source_model,
                              current_model,
                              optimizer,
                              groupLoss: Loss,
                              apex=False,
                              device=None):
    def _update(engine, batch):
        source_model.eval()
        current_model.train()
        optimizer.zero_grad()
        groupLoss.optimizer_zero_grad()

        # data
        img, cls_label = batch
        img = img.to(device)
        cls_label = cls_label.to(device)
        source_data = source_model(img)
        current_data = current_model(img)
        current_data.cls_label = cls_label
        current_data.source = source_data
        # train current module
        loss_values = {}
        loss = torch.tensor(0.0, requires_grad=True).to(device)
        for name, loss_fn in groupLoss.loss_function_map.items():
            loss_temp = loss_fn(current_data)
            loss += loss_temp
            loss_values[name] = loss_temp.item()

        if apex:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        groupLoss.optimizer_step()

        # compute acc
        acc = (current_data.cls_score.max(1)[1] == current_data.cls_label).float().mean()
        loss_values["Loss"] = loss.item()
        loss_values["Acc"] = acc.item()
        return loss_values

    return Engine(_update)
