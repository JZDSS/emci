import torch.nn as nn
import  torch


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()


    def forward(self, predictions, targets):
        return torch.mean((predictions - targets)**2)
