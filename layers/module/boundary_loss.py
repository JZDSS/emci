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

    def __init__(self, decay_threshold=0.1, threshold_decay_rate=0.99, mean_decay=0.99):
        super(BoundaryLoss, self).__init__()
        self.threshold = 1
        self.mean_loss = 0.1
        self.threshold_decay_rate = threshold_decay_rate
        self.decay_threshold = decay_threshold
        self.mean_decay = mean_decay
        self.k = 1 - mean_decay
        self.count = 0
        self.zero = None


    def forward(self, prediction, target):
        if self.zero is None:
            self.zero = torch.zeros_like(target)
        l1 = torch.abs(prediction - target)

        suppressed = torch.where(l1 < self.threshold, self.zero, l1).sum(dim=1).mean()
        return suppressed

    def update(self, loss):
        # if loss <=1e-3:
        #     self.threshold /= 2
        #     logging.info('threshold update: %e' % self.threshold)
        #     self.mean_loss = loss
        #     return
        self.mean_loss *= self.mean_decay
        self.mean_loss += self.k * loss
        if loss < self.mean_loss:
            self.count += 1
        else:
            self.count -= 1

        if self.count == 3:
            self.threshold *= self.threshold_decay_rate
            logging.info("threshold update: %e" % self.threshold)
            self.count = 0
        elif self.count == -1:
            self.threshold /= self.threshold_decay_rate
            logging.info("threshold update: %e" % self.threshold)
            self.count = 0

class BoundaryLoss2(nn.Module):
    def __init__(self, decay_threshold1, decay_rate1, mean_decay1,
                       decay_threshold2, decay_rate2, mean_decay2):
        super(BoundaryLoss2, self).__init__()
        self.bloss1 = BoundaryLoss(decay_threshold1, decay_rate1, mean_decay1)
        self.bloss2 = BoundaryLoss(decay_threshold2, decay_rate2, mean_decay2)

    def forward(self, prediction, target):
        loss1 = self.bloss1(prediction[:, :66], target[:, :66])
        loss2 = self.bloss2(prediction[:, 66:], target[:, 66:])
        return loss1 + loss2

class BoundaryLossN(nn.Module):
    def __init__(self):
        super(BoundaryLossN, self).__init__()
        self.bloss = []
        for i in range(106):
            self.bloss.append(BoundaryLoss())

    def forward(self, prediction, target):
        loss = 0
        for j, i in enumerate(list(range(0, 212, 2))):
            loss += self.bloss[j](prediction[:, i:i + 2])

        return loss
