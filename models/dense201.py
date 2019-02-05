import torch.nn as nn
from torchvision.models import densenet
import torch.utils.model_zoo as model_zoo
import re


class Dense201(densenet.DenseNet):
    def __init__(self, pretrained=True, num_classes=212, **kwargs):
        super(Dense201, self).__init__(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                         num_classes=num_classes, **kwargs)
        if pretrained:
            pretrained_dict = dict(model_zoo.load_url(densenet.model_urls['densenet201']))
            print(pretrained_dict.keys())
            del pretrained_dict['classifier.weight']
            del pretrained_dict['classifier.bias']
            model_dict = self.state_dict()
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(pretrained_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    pretrained_dict[new_key] = pretrained_dict[key]
                    del pretrained_dict[key]

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = super(Dense201, self).forward(x)
        out = self.sigmoid(out)
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