from data.build import make_multi_data_loader
from config import cfg
from utils.logger import setup_logger

logger = setup_logger("reid_baseline", "/home/arron/PycharmProjects/reid-baseline/unit_test", 0)
make_multi_data_loader(cfg)
