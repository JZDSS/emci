# _*_ coding:utf-8 _*_

from __future__ import print_function, division
import csv
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import linecache
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()


root_dir='/home/lmy/Training_data/AFW'

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        super(FaceLandmarksDataset, self).__init__()
        self.root = root_dir
        self.transform = transform
        self.landmarks = [file for file in os.listdir(os.path.join(root_dir, "landmark")) if os.path.splitext(file)[-1] == ".txt"]

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        ldmkroot = os.path.join(self.root, "landmark")
        itemlist = os.listdir(ldmkroot)
        ldmks = [items for items in itemlist if os.path.splitext(items)[-1] == ".txt"]
        # filename = os.path.join(self.root, "landmark", items)
        imgname = os.path.splitext(ldmks[idx])[0]


        img_name = os.path.join(self.root, "picture", imgname)
        image = io.imread(img_name)
        ldmks_name = os.path.join(self.root, "landmark", ldmks[idx])
        ldmk = np.arange(212).reshape(-1, 2)
        ldmk = b.astype(np.float64)
        for num in range(2, 108):
            a = linecache.getline(ldmks_name, num)
            a = count.strip('\n').split(' ')
            a = np.array(a)
            ldmk[num-2] = a
            sample = {'image': image, 'landmarks': ldmk}

            if self.transform:
                sample = self.transform(sample)

        return sample


def main():
    FaceLandmarksDataset(root_dir, "")

if __name__=="__main__":
    main()