import torch.nn as nn
from models import densenet
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import re
import torch
from models.saver import Saver_


class Dense201(densenet.DenseNet):
    def __init__(self, pretrained=True, num_classes=212, ckpt=None, **kwargs):
        super(Dense201, self).__init__(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                         num_classes=1, **kwargs)
        if pretrained:
            pretrained_dict = dict(model_zoo.load_url(densenet.model_urls['densenet201']))
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
        self.fc1 = nn.Linear(1920, 66)
        self.fc2 = nn.Linear(1920, 146)
        if not ckpt is None:
            saver = Saver_(ckpt, 'model')
            saver.load_last_ckpt(self)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        f1 = out[:,:960]
        f2 = out[:,960:]
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