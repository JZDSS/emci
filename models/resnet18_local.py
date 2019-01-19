import torchvision.models as models
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.utils.model_zoo as model_zoo
import torch


class ResNet18(resnet.ResNet):

    def __init__(self, pretrained=True, num_classes=212, **kwargs):
        super(ResNet18, self).__init__(resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes * 2, **kwargs)
        if pretrained:
            pretrained_dict = dict(model_zoo.load_url(resnet.model_urls['resnet18']))
            del pretrained_dict['fc.weight']
            del pretrained_dict['fc.bias']
            model_dict = self.state_dict()
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        self.pred1 = nn.Conv2d(64, 64, 3)
        self.pred2 = nn.Conv2d(128, 128, 3)
        self.pred3 = nn.Conv2d(256, 256, 3)
        self.pred4 = nn.Conv2d(512, 512, 3)
        self.pred1_c = nn.Conv2d(64, 212 * 2, 3)
        self.pred2_c = nn.Conv2d(128, 212 * 2, 3)
        self.pred3_c = nn.Conv2d(256, 212 * 2, 3)
        self.pred4_c = nn.Conv2d(512, 212 * 2, 3)
        self.sigmoid = nn.Sigmoid()
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        out.append(self.pred1_c(self.pred1(x)))

        x = self.layer2(x)
        out.append(self.pred1_c(self.pred2(x)))

        x = self.layer3(x)
        out.append(self.pred1_c(self.pred3(x)))

        x = self.layer4(x)
        out.append(self.pred1_c(self.pred4(x)))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out.append(self.fc(x))

        res_list = []
        prob_list = []
        for p in out:
            res, prob = p.split([212, 212], dim=1)
            res = res.view((res.shape[0], res.shape[1], -1))
            prob = prob.view((prob.shape[0], prob.shape[1], -1))
            res_list.append(res)
            prob_list.append(prob)
        res = torch.cat(res_list, dim=-1)
        prob = torch.cat(prob_list, dim=-1)
        res = self.sigmoid(res)
        prob = self.soft_max(prob)
        out = res * prob
        out = out.sum(dim=-1)
        return out
