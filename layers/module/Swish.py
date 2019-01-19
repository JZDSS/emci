import torch.nn as nn


class Swish(nn.Module):
    """
    创建了一个新的激活函数swish
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * nn.Sigmoid(x)
        return x
