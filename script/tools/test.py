import sys

sys.path.append('.')

from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from script.tools.expand import main


def test(cfg, saver):
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    if cfg.MODEL.DEVICE is 'cuda':
        model = model.cuda()
    to_load = {'model': model}
    saver.to_save = to_load
    saver.load_checkpoint(is_best=True)
    inference(cfg, model, val_loader, num_query)


if __name__ == '__main__':
    cfg, saver = main(["TEST.IF_ON", True])
    test(cfg, saver)
