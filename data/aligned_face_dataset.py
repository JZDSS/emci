import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from data import utils


class AlignedFaceDataset(Dataset):
    def __init__(self, root_dir, bin_dir, phase='train', shape=(224, 224), bbox_sclae=(128, 128)):
        """
        :param root_dir: icme文件夹路径，见README
        :param bin_dir:  train或者valid文件夹路径，见README
        :param phase:
        :param transform:
        """
        super(AlignedFaceDataset, self).__init__()
        self.shape = shape
        self.phase = phase
        self.bbox_sclae = bbox_sclae

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
                if self.phase == 'train':
                    p = max_n / len(files)
                    # 万一出现max_n是某个pose数量的2倍以上
                    while p > 1:
                        file_list.append(i)
                        p -= 1
                    # 掷色子决定是否再次重采样
                    dice = np.random.uniform(0, 1)
                    if dice < p:
                        file_list.append(i)
                elif self.phase == 'eval':
                    file_list.append(i)
        # 打乱顺序
        shuffle(file_list)
        img_dir = os.path.join(root_dir, 'aligned/picture')
        landmark_dir = os.path.join(root_dir, 'aligned/landmark')
        self.images = [os.path.join(img_dir, f) for f in file_list]
        self.landmarks = [os.path.join(landmark_dir, f + '.txt') for f in file_list]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        landmark_path = self.landmarks[i]
        landmarks = utils.read_landmarks(landmark_path)
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.shape)
        landmarks[:, 0] /= self.bbox_sclae[0]
        landmarks[:, 1] /= self.bbox_sclae[1]
        if self.phase == 'train':
            image, landmarks = utils.random_flip(image, landmarks, 0.5)
            image = utils.random_gamma_trans(image, np.random.uniform(0.8, 1.2, 1))
            image = utils.random_color(image)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return image, np.reshape(landmarks, (-1))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = FaceDataset("/data/icme", "/data/icme/train")
    b = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=4))
    while True:
        images, landmarks = next(b)
        image = images[0, :]
        plt.imshow(image)
        plt.show()
        print(images, landmarks)
