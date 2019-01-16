# _*_ coding:utf-8 _*_

# 测试代码
from __future__ import print_function, division
import os
import os.path
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

root = "faces\AFW"

class face_dataset(Dataset):
    def __init__(self, root, phase, transform=None):
        super(face_dataset, self).__init__()
        self.root = root
        self.transform = transform

        images = [file for file in os.listdir(os.path.join(root, "picture")) if os.path.splitext(file)[-1] == ".jpg"]

        self.images = [os.path.join(root, "picture", file) for file in images]
        self.landmarks = [os.path.join(root, "landmark", file+".txt") for file in images]

        if phase == "train":
            pass
        elif phase == "val":
            pass
        else:
            pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):

        return io.imread(self.images[i]), self.read_landmark(self.landmarks[i])

    def read_landmark(self, path):
        with open(path) as f:
            ldmk = f.read()
        ldmk = np.array([[int(x.split('.')[0]) for x in co.split()] for co in ldmk.split('\n')[1:-1]]).reshape((-1, 2))
        return ldmk

    def show(self, i):
        image, landmrk = self.__getitem__(i)
        print(self.images[i])
        print(landmrk)
        show_landmark(image, landmrk)

def show_landmark(image, landmark):
    plt.imshow(image)
    plt.scatter(landmark[:, 0], landmark[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def main():
    face_dataset(root, "")

if __name__ == '__main__':
    main()