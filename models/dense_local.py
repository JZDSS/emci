from models import dense201
import torch.nn as nn
import torch.nn.functional as F
import torch

class DenseLocal(dense201.Dense201):
    def __init__(self, pretrained=True, num_classes=212, ckpt=None, **kwargs):
        super(DenseLocal, self).__init__(pretrained, num_classes, ckpt, **kwargs)
        self.local = nn.Conv2d(1920, 212, 1)
        self.prob = nn.Conv2d(1920, 212, 1)
        self.prob_global = nn.Linear(1920, 212)

    def forward(self, x):
        features = self.features(x)
        features = F.relu(features, inplace=True)

        coord_local = self.local(features)
        prob_local = self.prob(features)

        coord_local = F.sigmoid(coord_local).view(-1, 106, 2, 7, 7)
        prob_local = F.sigmoid(prob_local).view(-1, 106, 2, 7, 7)

        features = F.avg_pool2d(features, kernel_size=7, stride=1).view(features.size(0), -1)

        coord_global = self.classifier(features)
        prob_global = self.prob_global(features)

        coord_global = F.sigmoid(coord_global)
        prob_global = F.sigmoid(prob_global)

        if self.training:
            return coord_local, prob_local, coord_global, prob_global
        else:
            max_prob = torch.max(torch.max(prob_local, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
            mask = prob_local == max_prob
            coord_local = torch.where(mask, coord_local, torch.zeros_like(coord_local))
            prob_local = torch.where(mask, prob_local, torch.zeros_like(prob_local))

            coord_local *= prob_local
            coord_local = coord_local.sum(dim=-1).sum(dim=-1).view(-1, 212)
            prob_local = prob_local.sum(dim=-1).sum(dim=-1).view(-1, 212)
            coord_global *= prob_global
            out = (coord_global + coord_local) / (prob_global + prob_local)
            return out
