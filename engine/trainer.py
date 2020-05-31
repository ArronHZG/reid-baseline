import logging

import torch
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage

from engine.inference import Eval
from loss import Loss
from tools.component import TrainComponent

logger = logging.getLogger("reid_baseline.train")


def do_train(cfg,
             train_loader,
             valid_dict,
             tr_comp: TrainComponent,
             saver):
    # tb_log = TensorBoardXLog(cfg, saver.save_dir)

    trainer = create_supervised_trainer(tr_comp.model,
                                        tr_comp.optimizer,
                                        tr_comp.loss,
                                        device=cfg.MODEL.DEVICE,
                                        apex=cfg.APEX.IF_ON)

    evaler = Eval(valid_dict, cfg.MODEL.DEVICE)
    evaler.get_valid_eval_map(cfg, tr_comp.model)

    run(cfg, train_loader, tr_comp, saver, trainer, evaler)


class Run:
    def __init__(self, name):
        self.name = name

    def __call__(self, x):
        return x[self.name]


def create_supervised_trainer(model, optimizer, groupLoss: Loss,
                              apex=False,
                              device=None):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        groupLoss.optimizer_zero_grad()

        img, cls_label = batch
        img = img.to(device)
        data = model(img)

        data.cls_label = cls_label.to(device)
        loss_values = {}
        loss = torch.tensor(0.0, requires_grad=True).to(device)
        for name, loss_fn in groupLoss.loss_function_map.items():
            loss_temp = loss_fn(data)
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
        acc = (data.cls_score.max(1)[1] == data.cls_label).float().mean()
        loss_values["Loss"] = loss.item()
        loss_values["Acc"] = acc.item()
        return loss_values

    return Engine(_update)


def run(cfg, train_loader, tr_comp, saver, trainer, evaler, tb_log=None):
    device = cfg.MODEL.DEVICE

    saver.to_save = {'trainer': trainer,
                     'model': tr_comp.model}
    # 'optimizer': tr_comp.optimizer,
    # 'center_param': tr_comp.loss_center,
    # 'optimizer_center': tr_comp.optimizer_center}
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=cfg.SAVER.CHECKPOINT_PERIOD),
                              saver.train_checkpointer,
                              saver.to_save)
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
        tr_comp.loss.scheduler_step()
        tr_comp.scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.TRAIN.LOG_ITER_PERIOD))
    def log_training_loss(engine):
        message = f"Epoch[{engine.state.epoch}], " + \
                  f"Iteration[{engine.state.iteration}/{len(train_loader)}], " + \
                  f"Base Lr: {tr_comp.scheduler.get_last_lr()[0]:.2e}, "

        if tr_comp.loss.xent and tr_comp.loss.xent.learning_weight:
            message += f"xentWeight: {tr_comp.loss.xent.uncertainty.mean().item():.4f}, "

        for loss_name in engine.state.metrics.keys():
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
        logger.info(f"Valid - Epoch: {engine.state.epoch}")

        sum_result = evaler.eval_multi_dataset()

        if saver.best_result < sum_result:
            logger.info(f'Save best: {sum_result:.4f}')
            saver.save_best_value(sum_result)
            saver.best_checkpointer(engine, saver.to_save)
            saver.best_result = sum_result
        else:
            logger.info(f"Not best: {saver.best_result:.4f} > {sum_result:.4f}")
        logger.info('-' * 80)

    if tb_log:
        tb_log.attach_handler(trainer, tr_comp.model, tr_comp.optimizer)
    # self.tb_logger.attach(
    #     validation_evaluator,
    #     log_handler=ReIDOutputHandler(tag="valid", metric_names=["r1_mAP"], another_engine=trainer),
    #     event_name=Events.EPOCH_COMPLETED,
    # )
    trainer.run(train_loader, max_epochs=cfg.TRAIN.MAX_EPOCHS)
    if tb_log:
        tb_log.close()
