import torch
from torch import nn
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')

class BoundaryLoss(nn.Module):

    def __init__(self, decay_threshold=10, threshold_decay_rate=0.9, mean_decay=0.99):
        super(BoundaryLoss, self).__init__()
        self.threshold = 1
        self.mean_loss = 0
        self.threshold_decay_rate = threshold_decay_rate
        self.decay_threshold = decay_threshold
        self.mean_decay = mean_decay
        self.k = 1 - mean_decay

    def forward(self, prediction, target):
        l2 = (prediction - target) ** 2
        suppressed = torch.where(l2 <= self.threshold, torch.zeros_like(l2), l2).sum(dim=1).mean()
        self.mean_loss *= self.mean_decay
        self.mean_loss += self.k * suppressed
        if self.mean_loss < self.decay_threshold:
            self.threshold *= self.threshold_decay_rate
            logging.info("threshold update: %f" % self.threshold)
        return suppressed


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
