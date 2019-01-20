from data.lb_dataset import LBDataset
from torch.utils.data import DataLoader
from models.resnet18 import ResNet18
import torchvision.models as models
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.optim as opt
import torch
import numpy as np

net = ResNet18().cuda()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(140, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 72)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

mlp = MLP()
a = LBDataset("/home/orion/correctdata", "/home/orion/correctdata/train")

criterion = nn.L1Loss()
optimizer = opt.Adam(mlp.parameters(), lr=1e-3, weight_decay=5e-4)
batch_size = 4
batch_iterator = iter(DataLoader(a, batch_size,
                                 shuffle=True, num_workers=0))
running_loss = 0.0
for iteration in range(10000):
    images, landmarks = next(batch_iterator)
    images = images.cuda()
    landmarks = landmarks.cuda()
    with torch.no_grad():
        out = net.forward(images)
    pr = out.cpu().data.numpy()

    out = mlp(pr[0:70]).double()

    optimizer.zero_grad()
    loss2 = criterion(out, landmarks[70:106])
    loss2.backward()

    optimizer.step()
    loss1 = criterion(pr[0:70], landmarks[0:70])
    loss = loss1 + loss2
    running_loss += loss.item()
    if iteration % 100 == 99:
        print('%5d loss: %.3f' %
              (iteration + 1, running_loss / 100))
        running_loss = 0.0
