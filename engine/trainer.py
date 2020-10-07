import logging
import sys

sys.path.append('.')
sys.path.append('..')

from ignite.engine import Events
from ignite.handlers import Timer, ModelCheckpoint
from ignite.metrics import RunningAverage

from unit.supervisedComponent import TrainComponent, create_supervised_trainer
from engine.inference import eval_multi_dataset
from data import make_train_data_loader_with_expand, make_train_data_loader, make_multi_valid_data_loader
from utils import main, Run

logger = logging.getLogger("reid_baseline.train")


def run(cfg, train_loader, tr_comp, saver, trainer, valid_dict):
    # TODO resume

    # trainer = Engine(...)
    # trainer.load_state_dict(state_dict)
    # trainer.run(data)
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
        eval_multi_dataset(cfg, valid_dict, tr_comp)

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

    # 测试数据集
    valid_dict = make_multi_valid_data_loader(cfg, dataset_name)

    # 训练组件
    tr_comp = TrainComponent(cfg, num_classes)
    trainer = create_supervised_trainer(tr_comp)

    run(cfg, train_loader, tr_comp, saver, trainer, valid_dict)
