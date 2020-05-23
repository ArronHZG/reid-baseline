import abc

from torch import nn

from utils.data import Data


class Base(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(Base, self).__init__()

    @abc.abstractmethod
    def forward(self, x) -> Data:
        pass
