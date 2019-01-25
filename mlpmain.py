import torch.nn.functional as F
from torch import nn
from data.lb_dataset import LBDataset
from torch.utils.data import DataLoader
import torch.optim as opt
import torch

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(212, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 125)
        self.fc4 = nn.Linear(125, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


mlp = MLP()
a = LBDataset("/home/zhzhong/Desktop/correctdata", "/home/zhzhong/Desktop/correctdata/train")
batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=0))
criterion = nn.L1Loss()
optimizer = opt.Adam(mlp.parameters(), lr=1e-3, weight_decay=5e-4)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for iteration in range(10000):
        bbox, landmarks = next(batch_iterator)

        landmarks = landmarks  # .cuda
        out = mlp(landmarks).double()
        optimizer.zero_grad()
        loss = criterion(out, bbox)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if iteration % 100 == 99:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, iteration + 1, running_loss / 100))
            running_loss = 0.0

