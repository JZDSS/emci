# _*_ coding:utf-8 _*_

import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as opt
from shw.LdmkDataset import LdmkDataset
import os
from wing_loss import WingLoss

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # 前71个landmark点，网络要重写
        self.fc1 = nn.Linear(130, 600)
        self.fc2 = nn.Linear(600, 300)
        # self.fc3 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(300, 82)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x

mlp = MLP().double()

a = LdmkDataset(os.path.join("icme","data","landmark"))
batch_loader = DataLoader(a, batch_size=50, shuffle=True, num_workers=0)
# batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=0))

# criterion = nn.L1Loss()
# criterion = nn.MSELoss()
criterion = WingLoss(7, 2)

optimizer = opt.Adam(mlp.parameters(), lr=2e-3, weight_decay=5e-4)

for epoch in range(500):

    running_loss = 0.0
    iteration = 0
    for inputs, labels in iter(batch_loader):
        # print(inputs, labels)
        # inputs, labels = next(batch_iterator)
        out = mlp(inputs).double()
        # print(out.size(), labels.size())
        optimizer.zero_grad()
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('\r[%d, %5d]   loss: %.4f' %
                (epoch + 1, iteration + 1, running_loss / 100), end=" ")
        running_loss = 0.0
        iteration += 1