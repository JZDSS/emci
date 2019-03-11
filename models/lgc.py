import torch
import torch.nn as nn
from dropblock import DropBlock2D
import torch.nn.functional as F
import numpy as np
from scipy import signal

class ConvNormReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, norm=True, relu=True, drop_prob=0., drop_block=7):
        super(ConvNormReLU, self).__init__()

        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias=False))
        if norm:
            self.add_module('norm', nn.BatchNorm2d(out_channels))
        if relu:
            self.add_module('relu', nn.ReLU())
        if drop_prob > 0:
            self.add_module('drop', DropBlock2D(drop_prob, drop_block))

    def forward(self, x):
        new_features = super(ConvNormReLU, self).forward(x)
        return new_features

class LocalContext(nn.Module):
    def __init__(self):
        super(LocalContext, self).__init__()
        self.feat = nn.Sequential()

        self.feat.add_module('conv5_0', ConvNormReLU(3, 128, 5, padding=2, drop_prob=0.1))

        for i in range(1, 5):
            self.feat.add_module('conv5_%d' % i, ConvNormReLU(128, 128, 5, padding=2, drop_prob=0.1))

        for i in range(10):
            self.feat.add_module('conv3_%d' % i, ConvNormReLU(128, 128, 3, padding=1, drop_prob=0.1))

        self.conv = ConvNormReLU(128, 106, 1)
        self.linear = ConvNormReLU(128, 106, 1, norm=False, relu=False)

    def forward(self, x):
        f = self.feat(x)
        return self.conv(f), self.linear(f)


class GlobalContext(nn.Sequential):
    def __init__(self):
        super(GlobalContext, self).__init__()
        self.add_module('dilated_conv3_0', ConvNormReLU(212, 128, 3, padding=4, dilation=4, drop_prob=0.1))
        for i in range(1, 7):
            self.add_module('dilated_conv3_%d' % i, ConvNormReLU(128, 128, 3, padding=4, dilation=4, drop_prob=0.1))



def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

class LGC(nn.Module):

    def __init__(self):
        super(LGC, self).__init__()

        self.local_context = LocalContext()

        kernel = 0
        for i in range(1, 6):
            kernel += gkern(45, 2*i - 1)
        kernel /= 5.

        self.kernel = torch.from_numpy(kernel).view((1, 1, 45, 45)).repeat(106, 1, 1, 1).float().detach()
        # self.kernel = nn.Parameter(torch.randn(106, 1, 45, 45))
        self.global_context = GlobalContext()
        self.linear = ConvNormReLU(128, 106, 1, norm=False, relu=False)

    def forward(self, x):
        h, o1 = self.local_context(x)

        # kernel convolution
        h = F.conv2d(h, self.kernel, padding=20, groups=106)
        o1 = F.conv2d(o1, self.kernel, padding=20, groups=106)

        h = torch.cat((h, o1), dim=1)
        h = self.global_context(h)
        o2 = self.linear(h)
        if not self.training:
            return o1 + o2
        else:
            return o1, o2
