import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.utils.model_zoo as model_zoo
import torch
import layers.module.swish as sw


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,  stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = sw.Swish()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(resnet.ResNet):

    def __init__(self, pretrained=True, num_classes=212, **kwargs):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
        if pretrained:
            pretrained_dict = dict(model_zoo.load_url(resnet.model_urls['resnet18']))
            del pretrained_dict['fc.weight']
            del pretrained_dict['fc.bias']
            model_dict = self.state_dict()
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        self.sigmoid = nn.Sigmoid()

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
        x = self.fc(x)
        res = self.sigmoid(x)


        return res
