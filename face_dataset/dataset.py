import os
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from face_dataset import utils

class FaceDataset(Dataset):
    def __init__(self, root_dir, bin_dir, phase=None, transform=None):
        """
        :param root_dir: icme文件夹路径，见README
        :param bin_dir:  train或者valid文件夹路径，见README
        :param phase:
        :param transform:
        """
        super(FaceDataset, self).__init__()
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
        img_dir = os.path.join(root_dir, 'data/pictures')
        landmark_dir = os.path.join(root_dir, 'data/landmarks')
        bbox_dir = os.path.join(root_dir, 'bbox')
        self.images = [os.path.join(img_dir, f) for f in file_list]
        self.landmarks = [os.path.join(landmark_dir, f + '.txt') for f in file_list]
        self.bboxes = [os.path.join(bbox_dir, f + '.rect') for f in file_list]

        self.transform = transform
        if phase == "train":
            pass
        elif phase == "val":
            pass
        else:
            pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        bbox_path = self.bboxes[i]
        landmark_path = self.landmarks[i]
        bbox = utils.read_bbox(bbox_path)
        landmarks = utils.read_landmarks(landmark_path)
        landmarks = utils.norm_landmarks(landmarks, bbox)
        image = io.imread(img_path)
        return image, np.reshape(landmarks, (-1))

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
    FaceDataset("/data/icme", "/data/icme/train")

if __name__ == '__main__':
    main()