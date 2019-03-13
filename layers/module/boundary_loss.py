import torch
from torch import nn
import numpy as np
import logging
import sys
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


class BoundaryLoss(nn.Module):

    def __init__(self, threshold=5e-2, threshold_decay=0.99):
        super(BoundaryLoss, self).__init__()
        self.threshold = torch.nn.Parameter(torch.tensor(threshold).cuda())
        self.threshold.requires_grad = False
        self.threshold_decay = threshold_decay
        self.count = 0


    def forward(self, prediction, target):
        pass

    def update(self, loss):
        if loss < self.threshold.item() * 0.1:
            self.count += 1
        else:
            self.count -= 1

        if self.count == 3:
            self.threshold *= self.threshold_decay
            # logging.info("threshold update: %e" % self.threshold)
            self.count = 0
        elif self.count == -1:
            self.threshold /= self.threshold_decay
            # logging.info("threshold update: %e" % self.threshold)
            self.count = 0


class SoftBoundaryLoss(BoundaryLoss):
    def __init__(self, alpha=10, threshold=0.1, threshold_decay=0.99):
        super(SoftBoundaryLoss, self).__init__(threshold, threshold_decay)
        self.alpha = alpha
        self.c = alpha ** -(1 / alpha)
        self.d = (1 / self.alpha - 1)


    def forward(self, prediction, target):


        x = torch.abs(prediction - target)

        suppressed = self.threshold ** (1 - self.alpha) * x ** self.alpha / self.alpha
        loss = torch.where(x < self.threshold, suppressed, x + (1 / self.alpha - 1) * self.threshold).sum(dim=1).mean()

        self.update(loss.cpu().data.numpy())

        return loss

        # x = torch.abs(prediction - target)
        # k = self.c * self.threshold ** self.d
        # y = (k * x) ** self.alpha
        # v = (k * self.threshold) ** self.alpha
        # y = torch.where(x < self.threshold, y, x + v - self.threshold).sum(dim=1).mean()
        # self.update(y.cpu().data.numpy())
        # return y


class HardBoundaryLoss(BoundaryLoss):
    def __init__(self, threshold=5e-2, threshold_decay=0.99):
        super(HardBoundaryLoss, self).__init__(threshold, threshold_decay)
        self.zero = None

    def forward(self, prediction, target):
        if self.zero is None:
            self.zero = torch.zeros_like(target)
        x = torch.abs(prediction - target)
        y = torch.where(x < self.threshold, self.zero, x - self.threshold).sum(dim=1).mean()
        self.update(y.cpu().data.numpy())
        return y


class BoundaryLossN(nn.Module):
    def __init__(self, version='soft', **kwargs):
        super(BoundaryLossN, self).__init__()
        if version == 'soft':
            loss = SoftBoundaryLoss
        elif version == 'hard':
            loss = HardBoundaryLoss
        else:
            raise RuntimeError('Unknown loss type!!')
        bloss = []
        for i in range(106):
            bloss.append(loss(**kwargs))
        self.bloss = torch.nn.ModuleList(bloss)
    def forward(self, prediction, target):
        loss = 0
        for j, i in enumerate(list(range(0, 212, 2))):
            loss += self.bloss[j](prediction[:, i:i + 2], target[:, i:i+2])
        return loss

class BoundaryLossMN(nn.Module):

    def __init__(self, version, **kwargs):
        super(BoundaryLossMN, self).__init__()
        blossn = []
        for i in range(11):
            blossn.append(BoundaryLossN(version, **kwargs))
        self.blossn = torch.nn.ModuleList(blossn)

    def forward(self, prediction, target, pose):
        loss = 0
        for i in range(prediction.shape[0]):
            po, pr, ta = pose[i:i + 1], prediction[i:i + 1], target[i:i + 1]
            loss += self.blossn[po](pr, ta)
        return loss
