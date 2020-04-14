import sys

sys.path.append('.')
sys.path.append('..')
from engine.uda import do_uda
from tools import main

if __name__ == '__main__':
    cfg, saver = main(["TEST.IF_ON", True, "UDA.IF_ON", True])
    saver.best_result = 0
    do_uda(cfg, saver)
