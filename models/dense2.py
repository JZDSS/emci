import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

from torchvision.models import densenet

class DenseNet(nn.Module):

    def __init__(self, num_classes):
        super(DenseNet, self).__init__()

        self.d1 = densenet.DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                         num_classes=1)
        self.d2 = densenet.DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                         num_classes=1)

        sigmoid = nn.Sigmoid()

        self.fc1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(3840, 66)),
            ('sig', sigmoid)
        ]))

        self.fc2 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(3840, 146)),
            ('sig', sigmoid)
        ]))

    def forward(self, x):
        f1 = self.d1.features(x)
        f1 = F.relu(f1, inplace=True)
        f1 = F.avg_pool2d(f1, kernel_size=7, stride=1).view(f1.size(0), -1)

        f2 = self.d2.features(x)
        f2 = F.relu(f2, inplace=True)
        f2 = F.avg_pool2d(f2, kernel_size=7, stride=1).view(f2.size(0), -1)

        feat1 = torch.cat((f1, f2.detach()), dim=1)
        feat2 = torch.cat((f1.detach(), f2), dim=1)

        out1 = self.fc1(feat1)
        out2 = self.fc2(feat2)

        out = torch.cat((out1, out2), dim=1)
        return out

if __name__ == '__main__':
    net = DenseNet(1)
    a = torch.randn(1, 3, 224, 224)
    b = net(a)
