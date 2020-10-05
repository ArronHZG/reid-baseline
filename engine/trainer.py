import itertools
import logging

import torch
from apex import amp
from ignite.engine import Engine, Events
from ignite.handlers import Timer, ModelCheckpoint
from ignite.metrics import RunningAverage
from torch import optim
from yacs.config import CfgNode

from data import make_train_data_loader_with_expand, make_train_data_loader, make_multi_valid_data_loader, OrderedDict
from engine.inference import Eval
from loss import MyCrossEntropy, ArcfaceLoss, TripletLoss, CenterLoss
from modeling import build_model
from solver import WarmupMultiStepLR
from utils import main, BaseComponent, Run, Data

logger = logging.getLogger("reid_baseline.train")


class TrainComponent(BaseComponent):
    def __init__(self, cfg: CfgNode, num_classes):
        super().__init__()
        self.model = build_model(cfg, num_classes).cuda()

        # loss
        self.loss_type = cfg.LOSS.LOSS_TYPE
        self.loss_function_map = OrderedDict()

        # ID loss
        if 'softmax' in self.loss_type:
            self.xent = MyCrossEntropy(num_classes=num_classes,
                                       label_smooth=cfg.LOSS.IF_LABEL_SMOOTH,
                                       learning_weight=cfg.LOSS.IF_LEARNING_WEIGHT).cuda()

            def loss_function(data: Data):
                return cfg.LOSS.ID_LOSS_WEIGHT * self.xent(data.cls_score, data.cls_label)

            self.loss_function_map["softmax"] = loss_function

        if 'arcface' in self.loss_type:
            self.arcface = ArcfaceLoss(num_classes=num_classes, feat_dim=feat_dim).cuda()

            def loss_function(data: Data):
                return self.arcface(data.feat_c, data.cls_label)

            self.loss_function_map["arcface"] = loss_function

        # metric loss
        if 'triplet' in self.loss_type:
            self.triplet = TripletLoss(cfg.LOSS.MARGIN,
                                       learning_weight=False).cuda()

            def loss_function(data: Data):
                return cfg.LOSS.METRIC_LOSS_WEIGHT * self.triplet(data.feat_t, data.cls_label)

            self.loss_function_map["triplet"] = loss_function

        # cluster loss
        if 'center' in self.loss_type:
            self.center = CenterLoss(num_classes=num_classes,
                                     feat_dim=self.model.in_planes,
                                     loss_weight=cfg.LOSS.CENTER_LOSS_WEIGHT,
                                     learning_weight=False).cuda()

            def loss_function(data: Data):
                return cfg.LOSS.CENTER_LOSS_WEIGHT * self.center(data.feat_t, data.cls_label)

            self.loss_function_map["center"] = loss_function

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=cfg.OPTIMIZER.BASE_LR,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

        self.xent_optimizer = optim.SGD(self.xent.parameters(),
                                        lr=cfg.OPTIMIZER.LOSS_LR / 5,
                                        momentum=cfg.OPTIMIZER.MOMENTUM,
                                        weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

        self.center_optimizer = optim.SGD(self.center.parameters(),
                                          lr=cfg.OPTIMIZER.LOSS_LR,
                                          momentum=cfg.OPTIMIZER.MOMENTUM,
                                          weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

        model_list = [self.model, self.xent, self.triplet, self.center]
        opt_list = [self.optimizer, self.center_optimizer, self.xent_optimizer]
        model_list, opt_list = amp.initialize(model_list,
                                              opt_list,
                                              opt_level='O1',
                                              loss_scale=1.0)
        self.model, self.xent, self.triplet, self.center = model_list[0], model_list[1], model_list[2], model_list[3]
        self.optimizer, self.center_optimizer, self.xent_optimizer = opt_list[0], opt_list[1], opt_list[2]

        self.scheduler = WarmupMultiStepLR(self.optimizer,
                                           cfg.WARMUP.STEPS,
                                           cfg.WARMUP.GAMMA,
                                           cfg.WARMUP.FACTOR,
                                           cfg.WARMUP.MAX_EPOCHS,
                                           cfg.WARMUP.METHOD)

    def __str__(self):
        s = f"{self.model}\n{self.loss_function_map.keys()}\n{self.optimizer}\n{self.scheduler}"
        return s


def create_supervised_trainer(tr_comp: TrainComponent):
    def _update(engine, batch):
        tr_comp.model.train()
        tr_comp.optimizer.zero_grad()
        tr_comp.center_optimizer.zero_grad()
        tr_comp.xent_optimizer.zero_grad()

        img, cls_label = batch
        img = img.to(torch.float).cuda()
        cls_label = cls_label.cuda()
        data = tr_comp.model(img)
        data.cls_label = cls_label

        loss_values = {}
        loss = torch.tensor(0.0, requires_grad=True).cuda()
        for name, loss_fn in tr_comp.loss_function_map.items():
            loss_temp = loss_fn(data)
            loss += loss_temp
            loss_values[name] = loss_temp.item()

        with amp.scale_loss(loss, tr_comp.optimizer) as scaled_loss:
            scaled_loss.backward()

        if 'center' in tr_comp.loss_type:
            for param in tr_comp.center.parameters():
                param.grad.data *= (1. / tr_comp.center.loss_weight)
        tr_comp.optimizer.step()
        tr_comp.center_optimizer.step()
        tr_comp.xent_optimizer.step()

        # compute acc
        acc = (data.cls_score.max(1)[1] == data.cls_label).float().mean()
        loss_values["Loss"] = loss.item()
        loss_values["Acc"] = acc.item()
        return loss_values

    return Engine(_update)


def run(cfg, train_loader, tr_comp, saver, trainer, evaler):
    # TODO resume

    # checkpoint
    handler = ModelCheckpoint(saver.model_dir, 'train', n_saved=3, create_dir=True)
    checkpoint_params = tr_comp.state_dict()
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              handler,
                              checkpoint_params)

    timer = Timer(average=True)
    timer.attach(trainer,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED,
                 step=Events.ITERATION_COMPLETED)
    # average metric to attach on trainer
    names = ["Acc", "Loss"]
    names.extend(tr_comp.loss_function_map.keys())
    for n in names:
        RunningAverage(output_transform=Run(n)).attach(trainer, n)

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        tr_comp.scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.TRAIN.LOG_ITER_PERIOD))
    def log_training_loss(engine):
        message = f"Epoch[{engine.state.epoch}], " + \
                  f"Iteration[{engine.state.iteration}/{len(train_loader)}], " + \
                  f"Base Lr: {tr_comp.scheduler.get_last_lr()[0]:.2e}, "

        for loss_name in engine.state.metrics.keys():
            message += f"{loss_name}: {engine.state.metrics[loss_name]:.4f}, "

        if tr_comp.xent and tr_comp.xent.learning_weight:
            message += f"xentWeight: {tr_comp.xent.uncertainty.mean().item():.4f}, "

        logger.info(message)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 80)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.EVAL.EPOCH_PERIOD))
    def log_validation_results(engine):
        logger.info(f"Valid - Epoch: {engine.state.epoch}")
        sum_result = evaler.eval_multi_dataset()
        logger.info('-' * 80)

    trainer.run(train_loader, max_epochs=cfg.TRAIN.MAX_EPOCHS)


if __name__ == '__main__':
    # 配置文件
    cfg, saver = main()

    # 数据集
    dataset_name = [cfg.DATASET.NAME]
    if cfg.JOINT.IF_ON:
        for name in cfg.JOINT.DATASET_NAME:
            dataset_name.append(name)
        train_loader, num_classes = make_train_data_loader_with_expand(cfg, dataset_name)
    else:
        train_loader, num_classes = make_train_data_loader(cfg, dataset_name[0])

    valid_dict = make_multi_valid_data_loader(cfg, dataset_name)

    # 训练组件
    tr_comp = TrainComponent(cfg, num_classes)

    trainer = create_supervised_trainer(tr_comp)
    evaler = Eval(valid_dict)
    evaler.get_valid_eval_map(cfg, tr_comp.model)

    run(cfg, train_loader, tr_comp, saver, trainer, evaler)
