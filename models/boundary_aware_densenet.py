import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers.module.fmf import FMF
from torchvision.models.densenet import _Transition, _DenseBlock


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    `"Look at Boundary: A Boundary-Aware Face Alignment Algorithm" <http://arxiv.org/abs/1805.10483>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), u_depth=[5, 4, 3],
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()
        # First convolution
        self.features = OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ])

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features['denseblock%d' % (i + 1)] = block
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features['transition%d' % (i + 1)] = trans
                num_features = num_features // 2
                fmf = FMF(num_features, num_features, u_depth[i])
                self.features['fmf%d' % (i + 1)] = fmf
        # Final batch norm
        self.features['norm5'] = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        for k, v in self.features.items():
            self.__setattr__(k, v)

    def forward(self, x, heatmap):
        features = x
        for k, v in self.features.items():
            print(k)
            if not 'fmf' in k:
                features = v(features)
            else:
                features = v(features, heatmap)

        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=8, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out



if __name__ == '__main__':
    import numpy as np
    # heatmap =np.zeros((128, 128, 3), np.uint8)
    heatmap = torch.FloatTensor(np.zeros((1, 13, 128, 128), np.uint8))
    # print(heatmap)
    # print(heatmap.shape)
    featuremap = torch.FloatTensor(np.random.random((1, 3, 256, 256)))
    #假设featuremap是64通道的32*32
    net = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                         num_classes=212)

    out = net(featuremap, heatmap)
    param = list(net.parameters())
    state = net.state_dict()
    a = 1
    #print(z.shape)
