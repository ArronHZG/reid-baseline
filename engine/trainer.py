import logging

import torch
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage

from tools.expand import TrainComponent
from utils.reid_metric import R1_mAP
from utils.tensorboardX_log import TensorBoardXLog


def create_supervised_trainer(model, optimizer, loss_fn_map,
                              apex=False,
                              device=None,
                              has_center=False,
                              center_criterion=None,
                              optimizer_center=None,
                              center_loss_weight=None):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        if has_center:
            optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device)
        target = target.to(device)
        feat, score = model(img)
        loss_values = {}
        loss = torch.tensor(.0, requires_grad=True).to(device)
        for name, loss_fn in loss_fn_map.items():
            loss_temp = loss_fn(score, feat, target)
            loss += loss_temp
            loss_values[name] = loss_temp.item()

        if apex:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if has_center:
            for param in center_criterion.parameters():
                param.grad.data *= (1. / center_loss_weight)
            optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        loss_values["Loss"] = loss.item()
        loss_values["Acc"] = acc.item()
        return loss_values

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(cfg,
             train_loader,
             val_loader,
             tr_comp: TrainComponent,
             num_query,
             saver):
    logger = logging.getLogger("reid_baseline.train")

    tb_log = TensorBoardXLog(cfg, saver.save_dir)

    device = cfg.MODEL.DEVICE

    trainer = create_supervised_trainer(tr_comp.model,
                                        tr_comp.optimizer,
                                        tr_comp.loss.loss_function_map,
                                        device=device,
                                        apex=cfg.APEX.IF_ON,
                                        has_center=cfg.LOSS.IF_WITH_CENTER,
                                        center_criterion=tr_comp.loss.center,
                                        optimizer_center=tr_comp.optimizer_center,
                                        center_loss_weight=cfg.LOSS.CENTER_LOSS_WEIGHT)

    saver.to_save = {'trainer': trainer,
                     'model': tr_comp.model,
                     'optimizer': tr_comp.optimizer,
                     'center_param': tr_comp.loss_center,
                     'optimizer_center': tr_comp.optimizer_center}

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=cfg.SAVER.CHECKPOINT_PERIOD),
                              saver.train_checkpointer,
                              saver.to_save)

    validation_evaluator = create_supervised_evaluator(tr_comp.model, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}, device=device)

    timer = Timer(average=True)
    timer.attach(trainer,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED,
                 step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x["Acc"]).attach(trainer, 'Acc')
    RunningAverage(output_transform=lambda x: x["Loss"]).attach(trainer, 'Loss')
    # for name in tr_comp.loss.loss_function_map.keys():
    #     RunningAverage(output_transform=lambda x: x[2][name]).attach(trainer, name)
    RunningAverage(output_transform=lambda x: x['softmax']).attach(trainer, 'softmax')
    RunningAverage(output_transform=lambda x: x['triplet']).attach(trainer, 'triplet')
    RunningAverage(output_transform=lambda x: x['center']).attach(trainer, 'center')

    # TODO start epoch
    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = 0

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        tr_comp.scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.TRAIN.LOG_ITER_PERIOD))
    def log_training_loss(engine):
        message = f"Epoch[{engine.state.epoch}], " + \
                  f"Iteration[{engine.state.iteration}/{len(train_loader)}], " + \
                  f"Base Lr: {tr_comp.scheduler.get_lr()[0]:.2e}, " + \
                  f"Loss: {engine.state.metrics['Loss']:.4f}, " + \
                  f"Acc: {engine.state.metrics['Acc']:.4f}, "

        for loss_name in tr_comp.loss.loss_function_map.keys():
            message += f"{loss_name}: {engine.state.metrics[loss_name]:.4f}, "

        logger.info(message)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 80)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.EVAL.EPOCH_PERIOD),
                saver=saver)
    def log_validation_results(engine, saver):
        # train_evaluator.run(train_loader)
        # cmc, mAP = validation_evaluator.state.metrics['r1_mAP']
        # logger.info("Train Results - Epoch: {}".format(engine.state.epoch))
        # logger.info("mAP: {:.1%}".format(mAP))
        # for r in [1, 5, 10]:
        #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

        validation_evaluator.run(val_loader)
        cmc, mAP = validation_evaluator.state.metrics['r1_mAP']
        logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        value = (mAP + cmc[0]) / 2

        if saver.best_result < value:
            logger.info(f'Save best: {value:.4f}')
            saver.save_best_value(value)
            saver.best_checkpointer(engine, saver.to_save)
            saver.best_result = value
        else:
            logger.info(f"Not best: {saver.best_result:.4f} > {value:.4f}")
        logger.info('-' * 80)

        if device == 'cuda':
            torch.cuda.empty_cache()

    tb_log.attach_handler(trainer, validation_evaluator, tr_comp.model, tr_comp.optimizer)

    trainer.run(train_loader, max_epochs=cfg.TRAIN.MAX_EPOCHS)

    tb_log.close()
