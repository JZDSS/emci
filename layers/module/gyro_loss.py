import torch
import torch.nn as nn
import numpy as np


class GyroLoss(nn.Module):

    def __init__(self, w, epsilon):
        super(GyroLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w * self.epsilon * self.epsilon - self.epsilon

    def forward(self, predictions, targets):
        """
        :param predictions: 网络输出，预测坐标，一个形状为[batch, N, 2]的张量
        :param targets: 目标，真实坐标，一个形状为[batch, N, 2]的张量
        :return: wing loss，标量
        """
        x = predictions - targets
        t = torch.abs(x)

        return torch.mean(torch.where(t < self.epsilon,
                                      self.w * x * x,
                                      t + self.C))
