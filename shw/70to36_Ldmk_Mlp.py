# _*_ coding:utf-8 _*_

import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as opt
from shw.LandmarkDataset import LdmkDataset
import os
from layers.module.wing_loss import WingLoss

import torch

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 前70个landmark点，网络要重写
        self.fc1 = nn.Linear(140, 500)
        self.fc2 = nn.Linear(500, 225)
        self.fc3 = nn.Linear(225, 125)
        self.fc4 = nn.Linear(125, 72)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


mlp = MLP().double()

# a = LBDataset("/home/zhzhong/Desktop/correctdata", "/home/zhzhong/Desktop/correctdata/train")

a = LdmkDataset(os.path.join("/home/orion/correctdata/data", "landmark"))

batch_loader = DataLoader(a, batch_size=1, shuffle=True, num_workers=0)
# batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=0))
# criterion = nn.L1Loss()
# criterion = nn.MSELoss()
criterion = WingLoss(10, 2)
optimizer = opt.Adam(mlp.parameters(), lr=2e-3, weight_decay=5e-4)
running_loss = 0.0
# for epoch in range(5000):  # loop over the dataset multiple times

iteration = 0
for inputs, labels in iter(batch_loader):

        # print(inputs, labels)
        # inputs, labels = next(batch_iterator)
        # landmarks = landmarks  # .cuda
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