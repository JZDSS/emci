from torch import nn
from models import dense201
import numpy as np
from data import utils
import torch
import matplotlib.pyplot as plt
import cv2

class TwoStage(nn.Module):
    def __init__(self, align):
        super(TwoStage, self).__init__()
        self.first = dense201.Dense201(ckpt='exp/wing2_5_13-bbox/snapshot')
        self.second = dense201.Dense201(ckpt='exp/wing2_5_13-align-j3-m0_2_0_1-2/snapshot')
        self.align = align
        self.shape = (224, 224)

    def forward(self, image, bbox):
        minx, miny, maxx, maxy = bbox
        im1 = image[miny:maxy + 1, minx:maxx + 1, :]
        im1 = cv2.resize(im1, self.shape)
        im1 = np.transpose(im1, (2, 0, 1)).astype(np.float32)
        im1 = torch.from_numpy(np.expand_dims(im1, 0)).cuda()
        pr1 = self.first(im1)

        pr1 = pr1.cpu().data.numpy()
        pr1 = utils.inv_norm_landmark(np.reshape(pr1, (-1, 2)), bbox)

        im2, _, t = self.align(image, pr1)
        im2 = np.transpose(im2, (2, 0, 1)).astype(np.float32)
        im2 = torch.from_numpy(np.expand_dims(im2, 0)).cuda()
        pr2 = self.second(im2)

        pr2 = pr2.cpu().data.numpy()
        pr2 = np.reshape(pr2, (-1, 2))
        pr2 *= im2.shape[-1]
        pr2 = self.align.inverse(pr2, t)
        return pr2

    # def forward(self, image):
        #
        # al_ldmk = self.first(image)[0]
        # image = image.cpu().data.numpy()[0]
        #
        # image = np.transpose(image, (1, 2, 0))
        #
        # al_ldmk = al_ldmk.view(-1, 2).cpu().data.numpy()
        # al_ldmk *= image.shape[1]
        #
        # image, _, t = self.align(image, al_ldmk)
        #
        # image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        # image = torch.from_numpy(image).cuda()
        #
        # pr = self.second(image).view((-1, 2)).cpu().data.numpy()
        # pr *= image.shape[-1]
        # pr = self.align.inverse(pr, t)
        # pr /= image.shape[-1]
        # return pr
