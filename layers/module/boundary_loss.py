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
        self.threshold = threshold
        self.threshold_decay = threshold_decay
        self.count = 0


    def forward(self, prediction, target):
        pass

    def update(self, loss):
        if loss < self.threshold * 0.1:
            self.count += 1
        else:
            self.count -= 1

        if self.count == 3:
            self.threshold *= self.threshold_decay
            logging.info("threshold update: %e" % self.threshold)
            self.count = 0
        elif self.count == -1:
            self.threshold /= self.threshold_decay
            logging.info("threshold update: %e" % self.threshold)
            self.count = 0


class SoftBoundaryLoss(BoundaryLoss):
    def __init__(self, alpha=100, threshold=5e-2, threshold_decay=0.99):
        super(SoftBoundaryLoss, self).__init__(threshold, threshold_decay)
        self.alpha = alpha
        self.c = alpha ** -(1 / alpha)
        self.d = (1 / self.alpha - 1)
        self.zero = None


    def forward(self, prediction, target):
        if self.zero is None:
            self.zero = torch.zeros_like(target)

        x = torch.abs(prediction - target)
        suppressed = torch.where(x < self.threshold, x, self.zero)
        k = self.c * self.threshold ** self.d
        y = (k * suppressed) ** self.alpha
        v = (k * self.threshold) ** self.alpha
        loss = torch.where(x < self.threshold, y, x + v - self.threshold).sum(dim=1).mean()
        self.update(y.cpu().data.numpy())

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
        self.bloss = []
        for i in range(106):
            self.bloss.append(loss(**kwargs))

    def forward(self, prediction, target):
        loss = 0
        for j, i in enumerate(list(range(0, 212, 2))):
            loss += self.bloss[j](prediction[:, i:i + 2], target[:, i:i+2])
        return loss
