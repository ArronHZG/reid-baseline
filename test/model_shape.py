import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


def count_param(model):
    """
    count module size
    :param model:
    :return:
    """
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def show_model(model, save_path, input_size=(3, 512, 512)):
    """

    :param model:
    :param model_name:
    :param input_size:
    :return:
    """
    # 模型打印
    summary(model, input_size, device="cpu")
    # model可视化
    writer = SummaryWriter(save_path)
    x = torch.rand(1, input_size[0], input_size[1], input_size[2])
    writer.add_graph(model, x)
    writer.close()
