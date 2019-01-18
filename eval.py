from data.face_dataset import FaceDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from layers.module.wing_loss import WingLoss
from models.saver import Saver
from models.resnet50 import ResNet50
from tensorboardX import SummaryWriter
import numpy as np
from utils.metrics import Metrics
import time
net = ResNet50()
sig = nn.Sigmoid()

#PATH = './ckpt'
a = FaceDataset("E:/correctdata", "E:/correctdata/valid")
batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=0))
#Saver.dir=PATH
saver = Saver('ckpt', 'model', 10)
current = saver.last_ckpt()
time.sleep(10)
net.eval()
batch_size = 4
epoch_size = len(a) // batch_size
writer = SummaryWriter('logs/wing_loss/valid')
metrics = Metrics().add_nme(0.99).add_auc().add_loss()
while True:
    if current == saver.last_ckpt():
        time.sleep(10)
    else:
        last = saver.last_ckpt()
        current_iter = 0 if last is None else int(last.split('.')[0].split('-')[-1])
        saver.load_last_ckpt(net)
        for iteration in range(11):
            if iteration % epoch_size == 0:
                batch_iterator = iter(DataLoader(a, batch_size,
                                                 shuffle=False, num_workers=0))
            images, landmarks = next(batch_iterator)

            with torch.no_grad():
                out = net.forward(images)
            out = sig(out)
            criterion = WingLoss(10, 0.5)
            loss = criterion(out, landmarks)
            pr = out.cpu().data.numpy()
            gt = landmarks.cpu().data.numpy()
            nme = metrics.nme.update(np.reshape(gt, (-1, 106, 2)), np.reshape(pr, (-1, 106, 2)))
            metrics.auc.update(nme)
            metrics.loss.update(loss.item())
        writer.add_scalar("watch/NME", metrics.nme.value * 100, current_iter)
        writer.add_scalar("watch/AUC", metrics.auc.value * 100, current_iter)
        writer.add_scalar("watch/LOSS", metrics.loss.value, current_iter)
        metrics.clear()
        pass




