import torch.nn
import torch

import torch
import torch.nn as nn
import numpy as np
from sklearn.externals import joblib

class DynamicWingLoss(nn.Module):

    def __init__(self):
        super(DynamicWingLoss, self).__init__()

    def forward(self, predictions, targets, w, epsilon):
        """
        :param predictions: 网络输出，预测坐标，一个形状为[batch, N, 2]的张量
        :param targets: 目标，真实坐标，一个形状为[batch, N, 2]的张量
        :return: wing loss，标量
        """
        x = predictions - targets
        t = torch.abs(x)
        C = w + w * np.log(1 + w / epsilon)
        return torch.mean(torch.where(t < w,
                           w * torch.log(1 + t / epsilon),
                           t - C))


class DynamicWingLoss2(nn.Module):

    def __init__(self):
        super(DynamicWingLoss2, self).__init__()
        self.dwing = DynamicWingLoss()

    def forward(self, predictions, targets, w1, epsilon1, w2, epsilon2):
        loss1 = self.dwing(predictions[:, :66], targets[:, :66], w1, epsilon1)
        loss2 = self.dwing(predictions[:, 66:], targets[:, 66:], w2, epsilon2)
        return loss1 + loss2

class PoseLoss(nn.Module):

    def __init__(self, max_w1, min_w1, max_epsilon1, min_epsilon1,
                       max_w2, min_w2, max_epsilon2, min_epsilon2):
        super(PoseLoss, self).__init__()
        self.max_w1, self.min_w1, self.max_epsilon1, self.min_epsilon1, \
        self.max_w2, self.min_w2, self.max_epsilon2, self.min_epsilon2 = \
            max_w1, min_w1, max_epsilon1, min_epsilon1,\
            max_w2, min_w2, max_epsilon2, min_epsilon2
        u = joblib.load('./cache/u.pkl')[:, 0]
        self.u = torch.from_numpy(u.astype(np.float32)).cuda()
        aligned = joblib.load('./cache/aligned.pkl')
        self.mean = torch.from_numpy(np.mean(aligned, axis=0, keepdims=True).astype(np.float32))
        self.mean = self.mean.cuda()
        self.dwing2 = DynamicWingLoss2()

    def forward(self, predictions, targets):
        """
        :param predictions: 网络输出，预测坐标，一个形状为[batch, N, 2]的张量
        :param targets: 目标，真实坐标，一个形状为[batch, N, 2]的张量
        :return: wing loss，标量
        """

        tmp = targets - self.mean
        pc = torch.abs(tmp @ self.u)

        loss = 0
        for i in range(targets.shape[0]):
            p = pc[i].cpu().data.numpy()
            w1 = (self.min_w1 - self.max_w1) * p + self.max_w1
            epsilon1 = (self.min_epsilon1 - self.max_epsilon1) * p + self.max_epsilon1
            w2 = (self.min_w2 - self.max_w2) * p + self.max_w2
            epsilon2 = (self.min_epsilon2 - self.max_epsilon2) * p + self.max_epsilon2
            loss += self.dwing2(predictions[i:i+1, :], targets[i:i+1, :], w1, epsilon1, w2, epsilon2)

        return loss / targets.shape[0]


