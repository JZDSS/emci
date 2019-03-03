import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from data import utils


class FaceDataset(Dataset):
    def __init__(self,
                 img_dir,
                 ldmk_dir,
                 bin_dir,
                 bins=[1,2,3,4,5,6,7,8,9,10,11],
                 phase='train',
                 shape=(224, 224),
                 img_format=None):
        """
        :param root_dir: icme文件夹路径，见README
        :param bin_dir:  train或者valid文件夹路径，见README
        :param phase:
        :param transform:
        """
        super(FaceDataset, self).__init__()
        self.shape = shape
        self.phase = phase
        bins = [str(bb) for bb in bins]
        # bin_dir为pdb.py中的图片输出目录（即cmd里的目录），root_dir为数据集根目录
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
        self.file_list = file_list
        self.images = [os.path.join(img_dir, f) for f in file_list]
        self.landmarks = [os.path.join(ldmk_dir, f + '.txt') for f in file_list]
        if img_format == 'png':
            self.images = [i.replace('.jpg', '.png') for i in self.images]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        while True:
            try:
                img_path = self.images[item]
                image = cv2.imread(img_path)
                landmark_path = self.landmarks[item]
                landmarks = utils.read_mat(landmark_path)
                return image, landmarks
            except:
                item += 1


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = FaceDataset("/data/icme/data/picture", '/data/icme/data/landmark', "/data/icme/train")
    b = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=4))
    while True:
        images, landmarks = next(b)
        image = images[0, :]
        plt.imshow(image)
        plt.show()
        print(images, landmarks)
