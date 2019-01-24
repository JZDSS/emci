# _*_ coding:utf-8 _*_

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from random import shuffle

root_dir = os.path.join('/home/orion/correctdata/data', 'landmark')

class LdmkDataset(Dataset):
    def __init__(self, root_dir):
        super(LdmkDataset, self).__init__()
        self.root = root_dir
        self.landmarks = [os.path.join(root_dir, x) for x in os.listdir(root_dir) if os.path.splitext(x)[-1] == ".txt"]

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, i):
        landmark = self.read_landmark(self.landmarks[i])
        landmark = landmark.astype(np.float)
        # print(landmark[105])
        x = (landmark[0:70]).flatten()
        y = (landmark[70:106]).flatten()
        # print(x.shape, y.shape)
        return torch.from_numpy(x).double(), torch.from_numpy(y).double()

    def read_landmark(self, path):
        with open(path) as f:
            ldmk = f.read()
        ldmk = np.array([[int(x.split('.')[0]) for x in co.split()] for co in ldmk.split('\n')[1:-1]]).reshape((-1, 2))
        return ldmk


if __name__ == '__main__':
    pass