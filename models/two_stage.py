from torch import nn
from models import dense201
import numpy as np
from data import utils
import torch

class TwoStage(nn.Module):
    def __init__(self, align):
        super(TwoStage, self).__init__()
        self.first = dense201.Dense201()
        self.second = dense201.Dense201()
        self.align = align

    def forward(self, image, bbox):

        al_ldmk = self.first(image)
        image = image.cpu().data.numpy()[0]

        image = np.transpose(image, (1, 2, 0))
        al_ldmk = al_ldmk.cpu().data.numpy()
        bbox = bbox.data.numpy()
        al_ldmk = utils.inv_norm_landmark(np.reshape(al_ldmk, (-1, 2)), bbox[0])

        image, _, t = self.aligner(image, al_ldmk)

        image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        image = torch.from_numpy(image).cuda()

        pre = self.second(image).view((-1, 2)).cpu().data.numpy()
        return pre
