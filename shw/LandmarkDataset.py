# _*_ coding:utf-8 _*_
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from random import shuffle
from data import utils


class FaceDataset(Dataset):
    def __init__(self, root_dir):
        super(FaceDataset, self).__init__()
        self.root = root_dir
        
        self.lanmarks

    def __len__(self):
        pass

    def __getitem__(self, i):
        pass



if __name__ == '__main__':
    pass
