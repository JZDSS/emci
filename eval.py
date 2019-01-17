from data.face_dataset import FaceDataset
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from layers.module.wing_loss import WingLoss
import utils
import numpy as np
net =models.resnet18(num_classes=212,pretrained=False)
sig = nn.Sigmoid()

PATH = './modelsave/90'
a = FaceDataset("E:/correctdata", "E:/correctdata/valid")
batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=0))


checkpoint =torch.load(PATH)
model = net
model.load_state_dict(checkpoint['net'])#optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()
batch_size = 4
epoch_size = len(a) // batch_size
Z=[]
Q=[]
for iteration in range(11):
    if iteration % epoch_size == 0:
        batch_iterator = iter(DataLoader(a, batch_size,
                                         shuffle=False, num_workers=0))
    images, landmarks = next(batch_iterator)

    with torch.no_grad():
        out = net.forward(images)
    out = sig(out)
    #print(utils.nme_calc(landmarks,out))

    criterion = WingLoss(10, 0.5)
    loss = criterion(out, landmarks)
    out = out.numpy()
    landmarks = landmarks.numpy()
    Z.append(out)
    Q.append(landmarks)

    if iteration % 10 == 0:
        #print(loss.item())
        X=np.stack(Z,axis=0)
        X = out.reshape(-1, 106, 2)
        Y = landmarks.reshape(-1, 106, 2)
        print(utils.auc_calc(utils.nme_calc(Y, X)))

