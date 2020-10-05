from unittest import TestCase

from engine.trainer import TrainComponent


class TestTrainComponent(TestCase):
    def test_TrainComponent(self):
        from config import cfg
        tc = TrainComponent(cfg, 1501)
        sd = tc.state_dict()
        print(sd)
        for item in sd.items():
            print(item[0])
