import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from data import utils
import cv2

class LBDataset(Dataset):
    def __init__(self, root_dir, bin_dir, phase='train', shape=(1, 212)):
        """
        :param root_dir: icme文件夹路径，见README
        :param bin_dir:  train或者valid文件夹路径，见README
        :param phase:
        :param transform:
        """
        super(LBDataset, self).__init__()
        self.shape = shape
        # bin_dir为pdb.py中的图片输出目录（即cmd里的目录），root_dir为数据集根目录
        bins = os.listdir(bin_dir)
        s = []
        for b in bins:
            curr = os.path.join(bin_dir, b)
            s.append(len(os.listdir(curr)))

        # 所有pose都采样到约max_n张，1.2倍是我随便设置的
        max_n = int(max(s) * 1.2)
        file_list = []
        for b in bins:
            curr = os.path.join(bin_dir, b)
            files = os.listdir(curr)
            for i in files:
                p = max_n / len(files)
                # 万一出现max_n是某个pose数量的2倍以上
                while p > 1:
                    file_list.append(i)
                    p -= 1
                # 掷色子决定是否再次重采样
                dice = np.random.uniform(0, 1)
                if dice < p:
                    file_list.append(i)
        # 打乱顺序
        shuffle(file_list)
        img_dir = os.path.join(root_dir, 'data/picture')
        landmark_dir = os.path.join(root_dir, 'data/landmark')
        bbox_dir = os.path.join(root_dir, 'bbox')
        self.images = [os.path.join(img_dir, f) for f in file_list]
        self.landmarks = [os.path.join(landmark_dir, f + '.txt') for f in file_list]
        self.bboxes = [os.path.join(bbox_dir, f + '.rect') for f in file_list]
        self.phase = phase

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, i):
        img_path = self.images[i]
        bbox_path = self.bboxes[i]
        landmark_path = self.landmarks[i]
        bbox = utils.read_bbox(bbox_path)
        landmarks = utils.read_mat(landmark_path)
        # landmarks = utils.norm_landmarks(landmarks, bbox)

        image = cv2.imread(img_path)
        bbox = np.array(bbox).astype(np.double)
        h, w, _ = image.shape
        landmarks[:,0] = landmarks[:,0] / h
        landmarks[:,1] = landmarks[:,1] / w
        bbox[[0, 2]] /= w
        bbox[[1, 3]] /= h
        # image = image[miny:maxy+1, minx:maxx+1, :]
        # image = cv2.resize(image, self.shape)
        #if self.phase == 'train':
            #image, landmarks = utils.random_flip(image, landmarks, 0.5)
            #image = utils.random_gamma_trans(image, np.random.uniform(0.8, 1.2, 1))
        #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return np.array(bbox).astype(np.double), np.reshape(landmarks, (-1))

