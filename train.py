import argparse
import numpy as np

import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

import cv2
from tensorboardX import SummaryWriter

from data.face_dataset import FaceDataset
from data.utils import draw_landmarks

from layers.module.wing_loss import WingLoss
from layers.module.gyro_loss import GyroLoss




parser = argparse.ArgumentParser(
    description='Landmark Detection Training')

parser.add_argument('-l', '--learning_rate', default=1e-3)
parser.add_argument('-b', '--batch_size', default=16)
parser.add_argument('-c', '--cuda', default=True)
parser.add_argument('-n', '--n_gpu', default=1)

args = parser.parse_args()

if __name__ == '__main__':
    writer = SummaryWriter('logs/wing_loss')
    net =models.resnet18(num_classes=212).cuda()
    sig = nn.Sigmoid()
    a = FaceDataset("/data/icme", "/data/icme/train")
    batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=4))

    criterion = WingLoss(10, 2)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)

    running_loss = 0.0
    batch_size = 8
    epoch_size = len(a) // batch_size
    for iteration in range(100000):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(a, batch_size,
                                                  shuffle=True, num_workers=4))
        # load train data
        images, landmarks = next(batch_iterator)
        images = images.cuda()
        landmarks = landmarks.cuda()

        out = net(images)
        out = sig(out)
        # backprop
        optimizer.zero_grad()
        loss = criterion(out, landmarks)
        loss.backward()
        optimizer.step()
        if iteration % 100 == 0:
            image = images.cpu().data.numpy()[0]
            gt = landmarks.cpu().data.numpy()[0]
            pr = out.cpu().data.numpy()[0]
            # 绿色的真实landmark
            image = draw_landmarks(image, gt, (0, 255, 0))
            # 红色的预测landmark
            image = draw_landmarks(image, pr, (0, 0, 255))
            image = image[::-1, ...]
            writer.add_image("result", image, iteration)
            writer.add_scalar("loss", loss.item(), iteration)
            writer.add_histogram("prediction", out.cpu().data.numpy(), iteration)
            state = {'net': net.state_dict(), 'iteration': iteration}  # 'optimizer': WingLoss.state_dict(),
            torch.save(state, './modelsave/%s' % iteration)
