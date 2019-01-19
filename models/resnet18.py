import torchvision.models as models
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.utils.model_zoo as model_zoo
import torch


class ResNet18(resnet.ResNet):

    def __init__(self, pretrained=True, num_classes=212, **kwargs):
        super(ResNet18, self).__init__(resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
        if pretrained:
            pretrained_dict = dict(model_zoo.load_url(resnet.model_urls['resnet18']))
            del pretrained_dict['fc.weight']
            del pretrained_dict['fc.bias']
            model_dict = self.state_dict()
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

        self.sigmoid = nn.Sigmoid()
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.sigmoid(x)
        return out
