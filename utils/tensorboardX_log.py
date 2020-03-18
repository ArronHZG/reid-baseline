import numbers
import os

import torch
from ignite.contrib.handlers.base_logger import BaseOutputHandler, BaseOptimizerParamsHandler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events


class ReIDOutputHandler(BaseOutputHandler):

    def __init__(self, tag, metric_names=None, output_transform=None, another_engine=None, global_step_transform=None):
        super(ReIDOutputHandler, self).__init__(tag, metric_names, output_transform, another_engine,
                                                global_step_transform)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with TensorboardLogger")

        metrics = self._setup_output_metrics(engine)

        global_step = self.global_step_transform(engine, event_name)

        if not isinstance(global_step, int):
            raise TypeError(
                "global_step must be int, got {}."
                " Please check the output of global_step_transform.".format(type(global_step))
            )

        for key, value in metrics.items():
            if isinstance(value, numbers.Number) or isinstance(value, torch.Tensor) and value.ndimension() == 0:
                logger.writer.add_scalars(f"{key}", {self.tag: value}, global_step)
            # elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
            #     for i, v in enumerate(value):
            #         logger.writer.add_scalar("{}/{}/{}".format(self.tag, key, i), v.item(), global_step)
            else:
                cmc, mAP = value
                logger.writer.add_scalars(f"metrics/mAP", {self.tag: mAP}, global_step)
                for r in [1, 5, 10]:
                    logger.writer.add_scalars(f"metrics/rank-{r}", {self.tag: cmc[r - 1]}, global_step)
                value = (mAP + cmc[0]) / 2
                logger.writer.add_scalars(f"metrics/mAP-rank-1", {self.tag: value}, global_step)


class OptimizerLearningRateHandler(BaseOptimizerParamsHandler):

    def __init__(self, optimizer, param_name="lr", tag=None):
        super(OptimizerLearningRateHandler, self).__init__(optimizer, param_name, tag)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'OptimizerParamsHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        logger.writer.add_scalar('lr', self.optimizer.param_groups[0], global_step)


class TensorBoardXLog:
    def __init__(self, cfg, log_dir):
        self.cfg = cfg
        work_dir = os.path.join(log_dir)
        self.tb_logger = TensorboardLogger(log_dir=os.path.join(work_dir))

    def close(self):
        self.tb_logger.close()

    def attach_handler(self, trainer, validation_evaluator, model, optimizer):
        if not self.cfg.TENSORBOARDX.IF_ON:
            return

        # self.tb_logger.attach(
        #     trainer,
        #     log_handler=OutputHandler(
        #         tag="training", output_transform=lambda loss, acc: {"acc": acc, "loss": loss}, metric_names=None
        #     ),
        #     event_name=Events.ITERATION_COMPLETED(every=100),
        # )

        # self.tb_logger.attach(
        #     train_evaluator,
        #     log_handler=ReIDOutputHandler(tag="train", metric_names=["loss", "r1_mAP"], another_engine=trainer),
        #     event_name=Events.EPOCH_COMPLETED,
        # )

        self.tb_logger.attach(
            validation_evaluator,
            log_handler=ReIDOutputHandler(tag="valid", metric_names=["r1_mAP"], another_engine=trainer),
            event_name=Events.EPOCH_COMPLETED,
        )

        self.tb_logger.attach(trainer,
                              log_handler=OptimizerParamsHandler(optimizer),
                              event_name=Events.EPOCH_COMPLETED)

        if self.cfg.TENSORBOARDX.SCALAR:
            self.tb_logger.attach(trainer, log_handler=WeightsScalarHandler(model), event_name=Events.EPOCH_COMPLETED)
            self.tb_logger.attach(trainer, log_handler=GradsScalarHandler(model), event_name=Events.EPOCH_COMPLETED)

        if self.cfg.TENSORBOARDX.HIST:
            self.tb_logger.attach(trainer, log_handler=WeightsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
            self.tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
