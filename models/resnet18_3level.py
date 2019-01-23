import torch
import torch.nn as nn
from models.resnet18 import ResNet18
import numpy as np

class SubNet(nn.Module):

    def __init__(self, pretrained=True, nets=[ResNet18, ResNet18, ResNet18], num_channels=[512, 512, 512], num_outputs=[30, 50, 26]):
        super(SubNet, self).__init__()
        assert len(nets) == len(num_outputs)
        assert sum(num_outputs) == 106

        add = np.zeros(len(num_outputs), dtype=np.int)
        for i in range(len(num_outputs)):
            add[i+1 :] += num_outputs[i]

        self.relu = nn.RReLU()

        self.nets = []
        self.fcs = []
        for i, n in enumerate(nets):
            self.nets.append(n(pretrained=pretrained, num_classes=num_channels[i]))
            self.fcs.append(nn.Linear(num_channels[i] + add[i]*2, num_outputs[i] * 2))

    def forward(self, input):

        feat = self.nets[0](self.relu(input))
        o = self.fcs[0](feat)

        for i in range(1, len(self.nets)):
            feat = self.nets[i](self.relu(input))
            feat = torch.cat([feat, o.detch()], dim=-1)
            curr_out = self.fcs[i](feat)
            o = torch.cat([o, curr_out], dim=-1)
        return o

a = SubNet()
a(torch.tensor(np.random.random((4, 3, 224, 224))).float())
