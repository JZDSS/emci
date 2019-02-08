import os
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.utils import draw_landmarks
from data.bbox_dataset import BBoxDataset
from data.align_dataset_input_fusion import AlignFusionDataset
from layers.module.wing_loss import WingLoss
from layers.module.gyro_loss import GyroLoss

from models.saver import Saver
from models.resnet50 import ResNet50
from models.resnet18 import ResNet18
from models.ba_dense201 import Dense201
from utils.metrics import Metrics
from utils.alignment import Align


parser = argparse.ArgumentParser(
    description='Landmark Detection Training')

parser.add_argument('-l', '--lr', default=2e-2, type=float)
parser.add_argument('-b', '--batch_size', default=16, type=int)
parser.add_argument('-c', '--cuda', default=True, type=bool)
parser.add_argument('-n', '--n_gpu', default=1, type=int)
parser.add_argument('-s', '--step', default=60000, type=int)
parser.add_argument('-g', '--gamma', default=0.2, type=float)
parser.add_argument('-w', '--weight_decay', default=5e-4, type=float)

args = parser.parse_args()

def adjust_learning_rate(optimizer, step, gamma, epoch, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # if epoch < 6:
    #     lr = 1e-8 + (args.lr-1e-8) * iteration / (epoch_size * 5)
    # else:
    #     lr = args.lr * (gamma ** ((iteration - epoch_size * 5) // step))
    lr = args.lr * (gamma ** (iteration // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    metrics = Metrics().add_nme(0.5).add_auc(decay=0.5).add_loss(decay=0.5)

    writer = SummaryWriter('logs/BA/train')
    net = Dense201().cuda()

    a = AlignFusionDataset('/data/icme/data/picture',
                 '/data/icme/data/landmark',
                 '/data/icme/data/landmark',
                 '/data/icme/train',
                 Align('../cache/mean_landmarks.pkl', (256, 256), (0.2, 0.05)))
    batch_iterator = iter(DataLoader(a, batch_size=args.batch_size, shuffle=True, num_workers=4))

    criterion = WingLoss(10, 2)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    saver = Saver('./ckpt', 'model', 10)
    last = saver.last_ckpt()
    start_iter = 0 if last is None else int(last.split('.')[0].split('-')[-1])
    if start_iter > 0:
        saver.load(net, last)
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

        lr = adjust_learning_rate(optimizer, args.step, args.gamma, epoch, iteration, epoch_size)
        # load train data
        images, landmarks, heatmap = next(batch_iterator)
        images = images.cuda()
        landmarks = landmarks.cuda()
        heatmap = heatmap.cuda()

        out = net(images, heatmap)
        # backprop
        optimizer.zero_grad()
        loss = criterion(out, landmarks)
        loss.backward()
        optimizer.step()
        if iteration % 100 == 0:
            image = images.cpu().data.numpy()[0]
            gt = landmarks.cpu().data.numpy()
            pr = out.cpu().data.numpy()
            image = image[45:48, :]
            # 绿色的真实landmark
            image = draw_landmarks(image, gt[0], (0, 255, 0))
            # 红色的预测landmark
            image = draw_landmarks(image, pr[0], (0, 0, 255))
            image = image[::-1, ...]
            nme = metrics.nme.update(np.reshape(gt, (-1, 106, 2)), np.reshape(pr, (-1, 106, 2)))
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
