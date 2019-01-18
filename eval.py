from data.face_dataset import FaceDataset
import torch
from torch.utils.data import DataLoader
from layers.module.wing_loss import WingLoss
from models.saver import Saver
from models.resnet50 import ResNet50
from models.resnet18 import ResNet18
from tensorboardX import SummaryWriter
import numpy as np
from utils.metrics import Metrics
import time
from data.utils import draw_landmarks


net = ResNet18().cuda()

criterion = WingLoss(10, 0.5)

#PATH = './ckpt'
a = FaceDataset("/data/icme", "/data/icme/valid")
batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=4))
#Saver.dir=PATH
saver = Saver('ckpt', 'model')
current = None
net.eval()
batch_size = 4
epoch_size = len(a) // batch_size
writer = SummaryWriter('logs/wing_loss/valid')
metrics = Metrics().add_nme(0.9).add_auc(decay=0.9).add_loss(decay=0.9)
while True:
    if current == saver.last_ckpt():
        time.sleep(1)
    else:
        last = saver.last_ckpt()
        current_iter = 0 if last is None else int(last.split('.')[0].split('-')[-1])
        while True:
            try:
                saver.load(net, last)
                break
            except:
                continue
        batch_iterator = iter(DataLoader(a, batch_size,
                                         shuffle=True, num_workers=0))
        images = None
        landmarks = None
        gt = None
        pr = None
        for iteration in range(10):
            images, landmarks = next(batch_iterator)
            images = images.cuda()
            landmarks = landmarks.cuda()
            with torch.no_grad():
                out = net.forward(images)

            loss = criterion(out, landmarks)
            pr = out.cpu().data.numpy()
            gt = landmarks.cpu().data.numpy()
            nme = metrics.nme.update(np.reshape(gt, (-1, 106, 2)), np.reshape(pr, (-1, 106, 2)))
            metrics.auc.update(nme)
            metrics.loss.update(loss.item())
        image = images.cpu().data.numpy()[0]
        # 绿色的真实landmark
        image = draw_landmarks(image, gt[0], (0, 255, 0))
        # 红色的预测landmark
        image = draw_landmarks(image, pr[0], (0, 0, 255))
        image = image[::-1, ...]
        writer.add_scalar("watch/NME", metrics.nme.value * 100, current_iter)
        writer.add_scalar("watch/AUC", metrics.auc.value * 100, current_iter)
        writer.add_scalar("watch/loss", metrics.loss.value, current_iter)
        writer.add_image("result", image, current_iter)
        metrics.clear()
        current = last





