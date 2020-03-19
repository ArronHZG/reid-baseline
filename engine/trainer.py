import logging

import torch
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP
from utils.tensorboardX_log import TensorBoardXLog


def create_supervised_trainer(model, optimizer, loss_fn, cfg,
                              device=None):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        feat, score = model(img)
        loss = loss_fn(score, feat, target)
        loss.backward()
        if cfg.APEX.IF_ON:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_with_center(model, optimizer, loss_fn, cfg,
                                          center_criterion=None,
                                          optimizer_center=None,
                                          center_loss_weight=None,
                                          device=None):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        feat, score = model(img)
        loss = loss_fn(score, feat, target)

        if cfg.APEX.IF_ON:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / center_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

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


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        saver,
        center_criterion=None,
        optimizer_center=None
):
    logger = logging.getLogger("reid_baseline")

    tb_log = TensorBoardXLog(cfg, saver.save_path)

    device = cfg.MODEL.DEVICE

    if center_criterion and optimizer_center:
        logger.info(f'With center loss, the loss type is {cfg.LOSS.LOSS_TYPE}')
        trainer = create_supervised_trainer_with_center(model,
                                                        optimizer,
                                                        loss_fn,
                                                        cfg,
                                                        center_criterion=center_criterion,
                                                        optimizer_center=optimizer_center,
                                                        center_loss_weight=cfg.LOSS.CENTER_LOSS_WEIGHT,
                                                        device=device)
        saver.to_save = {'trainer': trainer,
                         'model': model,
                         'optimizer': optimizer,
                         'center_param': center_criterion,
                         'optimizer_center': optimizer_center}

    else:
        logger.info(f'Without center loss, the loss type is {cfg.LOSS.LOSS_TYPE}')
        trainer = create_supervised_trainer(model,
                                            optimizer,
                                            loss_fn,
                                            cfg,
                                            device=device)
        saver.to_save = {'trainer': trainer,
                         'model': model,
                         'optimizer': optimizer}

    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        saver.load_checkpoint()

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=cfg.SAVER.CHECKPOINT_PERIOD),
                              saver.train_checkpointer,
                              saver.to_save)

    # train_evaluator = create_supervised_evaluator(model, metrics={
    #     'r1_mAP': R1_mAP(num_query, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}, device=device)

    validation_evaluator = create_supervised_evaluator(model, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, if_feat_norm=cfg.TEST.IF_FEAT_NORM)}, device=device)

    timer = Timer(average=True)
    timer.attach(trainer,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED,
                 step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.TRAIN.LOG_ITER_PERIOD))
    def log_training_loss(engine):

        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(engine.state.epoch, engine.state.iteration, len(train_loader),
                            engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                            scheduler.get_lr()[0]))

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
            logger.info('Save best')
            saver.save_best_value(value)
            saver.best_checkpointer(engine, saver.to_save)
        logger.info('-' * 80)

    tb_log.attach_handler(trainer, validation_evaluator, model, optimizer)

    trainer.run(train_loader, max_epochs=cfg.TRAIN.MAX_EPOCHS)

    tb_log.close()
