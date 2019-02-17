import os
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.utils import draw_landmarks
from data.bbox_dataset import BBoxDataset
from data.align_dataset import AlignDataset
from layers.module.wing_loss import WingLoss
from layers.module.wing_loss2 import WingLoss2
from layers.module.gyro_loss import GyroLoss

from models.saver import Saver
from models.resnet50 import ResNet50
from models.resnet18 import ResNet18
from models.dense201 import Dense201
from utils.metrics import Metrics
from utils.alignment import Align
from utils import learning_rate

parser = argparse.ArgumentParser(
    description='Landmark Detection Training')

parser.add_argument('-l', '--lr', default=1e-5, type=float)
parser.add_argument('-b', '--batch_size', default=16, type=int)
parser.add_argument('-c', '--cuda', default=True, type=bool)
parser.add_argument('-n', '--n_gpu', default=1, type=int)
parser.add_argument('-s', '--step', default=300, type=int)
parser.add_argument('-g', '--gamma', default=0.99, type=float)
parser.add_argument('-w', '--weight_decay', default=5e-4, type=float)

args = parser.parse_args()

# config = {0: learning_rate.polynomial_decay(1e-8, 8000, 2e-5),
#           8001: learning_rate.exponential_decay(2e-5, 500, 0.993)}
# lr_gen = learning_rate.mix(config)
lr_gen = learning_rate.piecewise_constant([100000, 150000], [2e-5, 4e-6, 8e-7])
# lr_gen = learning_rate.piecewise_constant([80000], [1e-5, 1e-6])
def adjust_learning_rate(optimizer):
    lr = lr_gen.get()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    metrics = Metrics().add_nme(0.5).add_auc(decay=0.5).add_loss(decay=0.5)

    writer = SummaryWriter('logs/wing2(5,15)-align-j3-margin-0_2-0_1_/train')
    net = Dense201(num_classes=212).cuda()
    # a = BBoxDataset('/data/icme/data/picture',
    #                 '/data/icme/data/landmark',
    #                 '/data/icme/bbox',
    #                 '/data/icme/train',
    #                 max_jitter=0)
    a = AlignDataset('/data/icme/data/picture',
                     '/data/icme/data/landmark',
                     '/data/icme/data/landmark',
                     '/data/icme/train',
                     Align('../cache/mean_landmarks.pkl', (224, 224), (0.2, 0.1),
                           ),# idx=list(range(51, 66))),
                     flip=True,
                     max_jitter=3,
                     max_radian=0
                     # ldmk_ids=list(range(51, 66))
                     )
    batch_iterator = iter(DataLoader(a, batch_size=args.batch_size, shuffle=True, num_workers=4))

    criterion = WingLoss2(5, 2, 15, 2)
    # criterion = WingLoss(10, 2)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    saver = Saver('../ckpt', 'model', 10)
    last = saver.last_ckpt()
    start_iter = 0 if last is None else int(last.split('.')[0].split('-')[-1])
    if start_iter > 0:
        saver.load(net, last)
        lr_gen.set_global_step(start_iter)

    running_loss = 0.0
    batch_size = args.batch_size
    epoch_size = len(a) // batch_size
    epoch = start_iter // epoch_size + 1
    for iteration in range(start_iter, 200001):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(a, batch_size,
                                                  shuffle=True, num_workers=4))
            epoch += 1

        lr = adjust_learning_rate(optimizer)
        # load train data
        images, landmarks = next(batch_iterator)
        images = images.cuda()
        landmarks = landmarks.cuda()

        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss = criterion(out, landmarks)
        loss.backward()
        optimizer.step()
        if iteration % 100 == 0:
            image = images.cpu().data.numpy()[0]
            gt = landmarks.cpu().data.numpy()
            pr = out.cpu().data.numpy()
            # 绿色的真实landmark
            image = draw_landmarks(image, gt[0], (0, 255, 0))
            # 红色的预测landmark
            image = draw_landmarks(image, pr[0], (0, 0, 255))
            image = image[::-1, ...]
            nme = metrics.nme.update(np.reshape(gt, (-1, gt.shape[1]//2, 2)), np.reshape(pr, (-1, gt.shape[1]//2, 2)))
            metrics.auc.update(nme)
            metrics.loss.update(loss)
            writer.add_scalar("watch/NME", metrics.nme.value * 100, iteration)
            writer.add_scalar("watch/AUC", metrics.auc.value * 100, iteration)
            writer.add_scalar("watch/loss", metrics.loss.value, iteration)
            writer.add_scalar("watch/learning_rate", lr, iteration)

            writer.add_image("result", image, iteration)
            writer.add_histogram("predictionx", out.cpu().data.numpy()[:, 0:212:2], iteration)
            state = net.state_dict()
            saver.save(state, iteration)

    torch.save(net.state_dict())
