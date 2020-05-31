import logging
import copy

import torch
from ignite.engine import Engine

from engine.inference import Eval
from engine.trainer import run, create_supervised_trainer
from loss import Loss
from tools.component import TrainComponent

logger = logging.getLogger("reid_baseline.ebll")

it = 0


def create_autoencoder_trainer(source_model,
                               current_model,
                               optimizer,
                               groupLoss: Loss,
                               apex=False,
                               device=None):
    def _update(engine, batch):
        global it
        it += 1
        source_model.eval()
        current_model.train()
        optimizer.zero_grad()
        # data
        img, _ = batch
        img = img.to(device)
        source_data = source_model(img)
        source_data.feat_t *= 100
        current_data = current_model(source_data.feat_t)
        current_data.ae = current_model
        current_data.feat_t = source_data.feat_t

        if it % 200 == 0:
            # print(data.ae)
            # for m in model.modules():
            #     if isinstance(m, nn.Linear):
            #         print(m.weight)
            code = current_data.ae.encoder(current_data.feat_t)
            print(f"\n\nfeat_t: {current_data.feat_t.size()} {current_data.feat_t.detach().cpu().numpy()[0][:5]}")
            print(f"code: {code.size()} {code.detach().cpu().numpy()[0][:5]}")
            print(f"recon_ae: {current_data.recon_ae.size()} {current_data.recon_ae.detach().cpu().numpy()[0][:5]}")
            print("\n\n")

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

        # compute acc
        loss_values["Loss"] = loss.item()
        loss_values["Acc"] = 0
        return loss_values

    return Engine(_update)


eb_it = 0


def create_ebll_trainer(source_model,
                        autoencoder,
                        current_model,
                        optimizer,
                        groupLoss: Loss,
                        apex=False,
                        device=None):
    def _update(engine, batch):
        global eb_it
        eb_it += 1
        source_model.eval()
        autoencoder.eval()
        current_model.train()

        optimizer.zero_grad()
        groupLoss.optimizer_zero_grad()

        # data
        img, cls_label = batch
        img = img.to(device)
        cls_label = cls_label.to(device)
        source_data = source_model(img)
        source_data.ae = autoencoder
        current_data = current_model(img)
        current_data.cls_label = cls_label
        current_data.source = source_data

        if eb_it % 200 == 0:
            # print(data.ae)
            # for m in model.modules():
            #     if isinstance(m, nn.Linear):
            #         print(m.weight)
            source_code = source_data.ae.encoder(source_data.feat_t)
            current_code = source_data.ae.encoder(current_data.feat_t)
            print("\n\n")
            print(f"code: {source_code.size()} {source_code.detach().cpu().numpy()[0][:5]}")
            print(f"code: {current_code.size()} {current_code.detach().cpu().numpy()[0][:5]}")
            print("\n\n")
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


def train_autoencoder(cfg,
                      train_loader,
                      valid_dict,
                      source_tr_comp: TrainComponent,
                      current_tr_comp: TrainComponent,
                      saver):
    trainer = create_autoencoder_trainer(source_tr_comp.model,
                                         current_tr_comp.model,
                                         current_tr_comp.optimizer,
                                         current_tr_comp.loss,
                                         device=cfg.MODEL.DEVICE,
                                         apex=cfg.APEX.IF_ON)

    evaler = Eval(valid_dict, cfg.MODEL.DEVICE)
    evaler.get_valid_eval_map_autoencoder(cfg, source_tr_comp.model, current_tr_comp.model)
    copy_cfg = copy.deepcopy(cfg)
    copy_cfg["TRAIN"]["MAX_EPOCHS"] = 90
    run(copy_cfg, train_loader, current_tr_comp, saver, trainer, evaler)


def fine_tune_current_model(cfg,
                            train_loader,
                            valid_dict,
                            tr_comp,
                            saver):
    for param in tr_comp.model.base.parameters():
        param.requires_grad = False
    trainer = create_supervised_trainer(tr_comp.model,
                                        tr_comp.optimizer,
                                        tr_comp.loss,
                                        device=cfg.MODEL.DEVICE,
                                        apex=cfg.APEX.IF_ON)

    evaler = Eval(valid_dict, cfg.MODEL.DEVICE)
    evaler.get_valid_eval_map(cfg, tr_comp.model)
    copy_cfg = copy.deepcopy(cfg)
    copy_cfg["TRAIN"]["MAX_EPOCHS"] = 30
    run(copy_cfg, train_loader, tr_comp, saver, trainer, evaler)


def ebll_train(cfg,
               train_loader,
               valid_dict,
               source_tr_comp,
               current_tr_comp,
               autoencoder_tr,
               saver):
    for param in current_tr_comp.model.base.parameters():
        param.requires_grad = True

    trainer = create_ebll_trainer(source_tr_comp.model,
                                  autoencoder_tr.model,
                                  current_tr_comp.model,
                                  current_tr_comp.optimizer,
                                  current_tr_comp.loss,
                                  apex=cfg.APEX.IF_ON,
                                  device=cfg.MODEL.DEVICE)

    evaler = Eval(valid_dict, cfg.MODEL.DEVICE)
    evaler.get_valid_eval_map_ebll(cfg, source_tr_comp.model, current_tr_comp.model)
    run(cfg, train_loader, current_tr_comp, saver, trainer, evaler)
    # sum_result = evaler.eval_multi_dataset()
