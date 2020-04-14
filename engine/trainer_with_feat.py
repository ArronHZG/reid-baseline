import logging

import torch
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage

from engine.inference import get_valid_eval_map, eval_multi_dataset
from engine.trainer import Run
from loss import Loss
from tools.expand import TrainComponent
from utils.tensorboardX_log import TensorBoardXLog

logger = logging.getLogger("reid_baseline.train")


def create_supervised_trainer(model, optimizer, loss_class: Loss,
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
        t_img, t_pid, s_img, s_target_feat = batch
        t_img = t_img.to(device)
        t_pid = t_pid.to(device)
        s_img = s_img.to(device)
        s_target_feat = s_target_feat.to(device)
        s_feat_t, s_feat_c, _ = model(s_img)
        t_feat_t, t_feat_c, t_score = model(t_img)

        loss_values = {}
        loss = torch.tensor(.0, requires_grad=True).to(device)
        for name, loss_fn in loss_class.loss_function_map.items():
            loss_temp = loss_fn(t_score, t_feat_t, t_pid)
            # loss += loss_temp
            loss_values[name] = loss_temp.item()

        feat_loss = loss_class.feat_loss(s_feat_c, s_target_feat)
        loss += feat_loss
        loss_values['feat'] = feat_loss.item()

        if apex:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        # if has_center:
        #     for param in center_criterion.parameters():
        #         param.grad.data *= (1. / center_loss_weight)
        #     optimizer_center.step()

        # compute acc
        acc = (t_score.max(1)[1] == t_pid).float().mean()
        loss_values["Loss"] = loss.item()
        loss_values["Acc"] = acc.item()
        return loss_values

    return Engine(_update)


def do_train_with_feat(cfg,
                       train_loader,
                       valid,
                       tr_comp: TrainComponent,
                       saver):
    tb_log = TensorBoardXLog(cfg, saver.save_dir)

    device = cfg.MODEL.DEVICE

    trainer = create_supervised_trainer(tr_comp.model,
                                        tr_comp.optimizer,
                                        tr_comp.loss,
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

    # multi-valid-dataset
    validation_evaluator_map = get_valid_eval_map(cfg, device, tr_comp.model, valid)

    timer = Timer(average=True)
    timer.attach(trainer,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED,
                 step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    names = ["Acc", "Loss"]
    names.extend(tr_comp.loss.loss_function_map.keys())

    for n in names:
        RunningAverage(output_transform=Run(n)).attach(trainer, n)

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
                  f"Lr: {tr_comp.scheduler.get_lr()[0]:.2e}, " + \
                  f"Loss: {engine.state.metrics['Loss']:.4f}, " + \
                  f"Acc: {engine.state.metrics['Acc']:.4f}, "

        for loss_name in tr_comp.loss.loss_function_map.keys():
            message += f"{loss_name}: {engine.state.metrics[loss_name]:.4f}, "

        message += f"feat: {engine.state.metrics['feat']:.4f}"

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

        logger.info(f"Valid - Epoch: {engine.state.epoch}")

        sum_result = eval_multi_dataset(device, validation_evaluator_map, valid)

        if saver.best_result < sum_result:
            logger.info(f'Save best: {sum_result:.4f}')
            saver.save_best_value(sum_result)
            saver.best_checkpointer(engine, saver.to_save)
            saver.best_result = sum_result
        else:
            logger.info(f"Not best: {saver.best_result:.4f} > {sum_result:.4f}")
        logger.info('-' * 80)

    tb_log.attach_handler(trainer, tr_comp.model, tr_comp.optimizer)

    # self.tb_logger.attach(
    #     validation_evaluator,
    #     log_handler=ReIDOutputHandler(tag="valid", metric_names=["r1_mAP"], another_engine=trainer),
    #     event_name=Events.EPOCH_COMPLETED,
    # )

    trainer.run(train_loader, max_epochs=cfg.TRAIN.MAX_EPOCHS)

    tb_log.close()
