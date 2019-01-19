import torch
import torch.nn as nn
import numpy as np


class WingLoss(nn.Module):

    def __init__(self, w, epsilon):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w + self.w * np.log(1 + self.w / self.epsilon)

    def forward(self, predictions, targets):
        """
        :param predictions: 网络输出，预测坐标，一个形状为[batch, N, 2]的张量
        :param targets: 目标，真实坐标，一个形状为[batch, N, 2]的张量
        :return: wing loss，标量
        """
        x = predictions - targets
        t = torch.abs(x)

        return torch.mean(torch.where(t < self.w,
                           self.w * torch.log(1 + t / self.epsilon),
                           t - self.C))


class Swish_act(nn.Module):
    """
    创建了一个新的激活函数swish
    """
    def __init__(self):
        super(Swish_act, self).__init__()

    def forward(self, x):
        x = x * nn.Sigmoid(x)
        return x
