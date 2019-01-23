from data.face_dataset import FaceDataset
from data.lb_dataset import LBDataset
from torch.utils.data import DataLoader
from torch import nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.optim as opt
import torch
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(140, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 72)


    def forward(self, x):
        x = F.relu(self.fc1(x)) #self.fc1(x)就是nn.Linear(140,800)(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

mlp = MLP()
a = LdmkDataset(os.path.join("/home/orion/correctdata/data", "landmark"))
batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=0))
# batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=0))
criterion = nn.MSELoss()
optimizer = opt.Adam(mlp.parameters(), lr=1e-3, weight_decay=5e-4)

running_loss = 0.0
for iteration in range(10000):
    inputs, labels = next(batch_iterator)
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
