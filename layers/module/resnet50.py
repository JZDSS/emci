import torchvision.models as models
import torch.nn  as nn

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.fc = nn.Linear(1000, 212)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return nn.Sigmoid(x)
