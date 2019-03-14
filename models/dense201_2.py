import torch.nn as nn
from models import dense201
import torch.nn.functional as F
import torch

class Dense201(nn.Module):
    def __init__(self, num_classes):
        super(Dense201, self).__init__()
        self.d1 = dense201.Dense201(num_classes=1)
        self.d2 = dense201.Dense201(num_classes=1)
        self.fc1 = nn.Linear(3840, 66)
        self.fc2 = nn.Linear(3840, 146)


    def forward(self, x):
        features = self.d1.features(x)
        out = F.relu(features, inplace=True)
        f1 = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)

        features = self.d2.features(x)
        out = F.relu(features, inplace=True)
        f2 = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)

        feat1 = torch.cat((f1.detach(), f2), dim=1)
        feat2 = torch.cat((f1, f2.detach()), dim=1)

        o1 = self.fc1(feat1)
        o2 = self.fc2(feat2)
        out = F.sigmoid(torch.cat((o1, o2), dim=1))
        return out

if __name__ == '__main__':
    import numpy as np
    import torch
    # heatmap =np.zeros((128, 128, 3), np.uint8)
    heatmap = torch.FloatTensor(np.zeros((1, 13, 128, 128), np.uint8))
    # print(heatmap)
    # print(heatmap.shape)
    featuremap = torch.FloatTensor(np.random.random((1, 3, 256, 256)))
    #假设featuremap是64通道的32*32
    net = Dense201(num_classes=212)

    out = net(featuremap)