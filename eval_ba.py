import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from data.bbox_dataset import BBoxDataset
import torch
from torch.utils.data import DataLoader
from layers.module.wing_loss import WingLoss
from models.saver import Saver
from models.resnet50 import ResNet50
from models.resnet18 import ResNet18
from models.ba_dense201 import Dense201
from tensorboardX import SummaryWriter
import numpy as np
from utils.metrics import Metrics
import time
from data.utils import draw_landmarks
from utils.alignment import Align
from data.bbox_dataset import BBoxDataset
from data.align_dataset import AlignDataset
from data.align_dataset_input_fusion import AlignFusionDataset

net = Dense201(num_classes=212).cuda()

criterion = WingLoss(10, 2)

#PATH = './ckpt'
# a = BBoxDataset('/data/icme/data/picture',
#                     '/data/icme/data/landmark',
#                     '/data/icme/bbox',
#                     '/data/icme/valid')
a = AlignFusionDataset('/data/icme/data/picture',
                 '/data/icme/data/landmark',
                 '/data/icme/data/pred_landmark',
                 '/data/icme/valid',
                 Align('../cache/mean_landmarks.pkl', (256, 256), (0.15, 0.1),
                       ), # idx=list(range(51, 66))),
                 phase='eval',
                 shape=(256, 256),
                 # ldmk_ids=list(range(51, 66))
                 )
batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=4))
#Saver.dir=PATH
saver = Saver('../ckpt', 'model')
current = None
net.eval()
batch_size = 4
epoch_size = len(a) // batch_size
writer = SummaryWriter('logs/BA/valid')
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
        heatmap = None
        gt = None
        pr = None
        for iteration in range(4):
            images, landmarks, heatmap = next(batch_iterator)
            images = images.cuda()
            landmarks = landmarks.cuda()
            heatmap = heatmap.cuda()
            with torch.no_grad():
                out = net.forward(images, heatmap)

            loss = criterion(out, landmarks)
            pr = out.cpu().data.numpy()
            gt = landmarks.cpu().data.numpy()
            nme = metrics.nme.update(np.reshape(gt, (-1, gt.shape[1]//2, 2)), np.reshape(pr, (-1, gt.shape[1]//2, 2)))
            metrics.auc.update(nme)
            metrics.loss.update(loss.item())
        image = images.cpu().data.numpy()[0]
        image = image[-3:, :]
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





