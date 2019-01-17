import torchvision.models as models
import torch.nn as nn


class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.fc = nn.Linear(1000, 212)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.model(x)
        h = self.fc(h)
        return self.sigmoid(h)
