import torchvision.models as models
import torch.nn  as nn
import torch

class Refined_resnet(nn.Module):
    def __init__(self, x):
        super(Refined_resnet, self).__init__()
        self.model = self.refined_resnet()
        self.output = self.forward(x)

    def refined_resnet(self):
        model = models.resnet50(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 212)
        model.add_module('Sigmoid', nn.Sigmoid())
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def forward(self, x):
        return self.model(x)
