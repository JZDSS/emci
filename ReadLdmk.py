## _*_ coding:utf-8 _*_

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch


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

        x = (np.vstack((landmark[0:33],landmark[33],landmark[37:39],landmark[42],landmark[46],landmark[50:58],landmark[63:67],
                        landmark[70],landmark[75],landmark[79],landmark[84],landmark[87],landmark[90],landmark[93]))).flatten()
        y = (np.vstack((landmark[34:37],landmark[39:42],landmark[43:46],landmark[47:50],landmark[58:63],landmark[67:70],landmark[71:75],
                        landmark[76:79],landmark[80:84],landmark[85:87],landmark[88:90],landmark[91:93],landmark[94:106]))).flatten()

        # x = (landmark[0:71]).flatten()
        # y = (landmark[71:106]).flatten()

        # print(x.shape, y.shape)
        return torch.from_numpy(x).double(), torch.from_numpy(y).double()

    def read_landmark(self, path):
        with open(path) as f:
            ldmk = f.read()
        ldmk = np.array([[int(x.split('.')[0]) for x in co.split()] for co in ldmk.split('\n')[1:-1]]).reshape((-1, 2))
        return ldmk


if __name__ == '__main__':
    pass