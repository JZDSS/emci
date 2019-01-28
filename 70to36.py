# _*_ coding:utf-8 _*_

from data.face_dataset import FaceDataset
from data.lb_dataset import LBDataset
from torch.utils.data import DataLoader
from torch import nn
from ReadLdmk import LdmkDataset
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.optim as opt
import torch
import numpy as np
import os
from layers.module.wing_loss import WingLoss

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 前70个landmark点，网络要重写
        self.fc1 = nn.Linear(114, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 98)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


mlp = MLP().double()
a = LdmkDataset(os.path.join("/home/orion/correctdata/data", "landmark"))
batch_loader = DataLoader(a, batch_size=4, shuffle=True, num_workers=0)
# criterion = nn.L1Loss()
# criterion = nn.MSELoss()
criterion = WingLoss(5, 2)
optimizer = opt.Adam(mlp.parameters(), lr=2e-3, weight_decay=5e-4)

for epoch in range(500):   # loop over the dataset multiple times
    running_loss = 0.0
    iteration = 0
    for inputs, labels in iter(batch_loader):
        # print(inputs, labels)
        out = mlp(inputs).double()
        # print(out.size(), labels.size())

        optimizer.zero_grad()
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if iteration % 100 == 99:
            print('\r%5d loss: %.3f' %
                  (iteration + 1, running_loss / 100), end="\n")
            running_loss = 0.0
        iteration += 1