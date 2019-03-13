import os
import cv2
import numpy as np
from data.face_dataset import FaceDataset
from data import utils

class BBoxDataset(FaceDataset):

    def __init__(self,
                 img_dir,
                 ldmk_dir,
                 bin_dir,
                 bins=[1,2,3,4,5,6,7,8,9,10,11],
                 phase='train',
                 shape=(224, 224),
                 max_jitter=0,
                 max_angle=0):
        super(BBoxDataset, self).__init__(img_dir, ldmk_dir, bin_dir, bins, phase, shape)
        # self.bboxes = [os.path.join(bbox_dir, f + '.rect') for f in self.file_list]
        self.max_jitter = max_jitter
        self.max_rand = max_angle / 180 * np.pi
        self.idx = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                    11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36,
                    35, 34, 33, 41, 40, 39, 38, 51, 52, 53, 54, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56,
                    55, 79, 78, 77, 76, 75, 82, 81, 80, 83, 70, 69, 68, 67, 66, 73, 72, 71, 74, 90, 89,
                    88, 87, 86, 85, 84, 95, 94, 93, 92, 91, 100, 99, 98, 97, 96, 103, 102, 101, 105, 104]
    def __getitem__(self, item):
        image, landmarks = super(BBoxDataset, self).__getitem__(item)
        H, W, _ = image.shape
        resize = \
            np.array([
                [self.shape[0] / W, 0, 0],
                [0, self.shape[1] / H, 0],
                [0, 0, 1]],
                dtype=np.float32)
        if self.phase == 'train':
            theta = np.random.uniform(-self.max_rand, self.max_rand)
            dx = np.random.uniform(-self.max_jitter, self.max_jitter)
            dy = np.random.uniform(-self.max_jitter, self.max_jitter)
            dice = np.random.uniform(0, 1)
            if dice > 0.5:
                flip = \
                    np.array([
                        [1, 0, self.shape[1]],
                        [0, 1, 0],
                        [0, 0, 1]],
                        dtype=np.float32
                    ) @ \
                    np.array([
                        [-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]],
                        dtype=np.float32)
                landmarks = landmarks[self.idx]
            else:
                flip = np.identity(3, np.float32)
            center = \
                np.array([
                    [1, 0, -self.shape[1] / 2],
                    [0, 1, -self.shape[1] / 2],
                    [0, 0, 1]],
                    dtype=np.float32
                )

            c = np.cos(theta)
            s = np.sin(theta)
            rotation = \
                np.array([
                    [c, s, 0],
                    [-s, c, 0],
                    [0, 0, 1]],
                    dtype=np.float32
                )

            decenter = \
                np.array([
                    [1, 0, self.shape[1] / 2],
                    [0, 1, self.shape[1] / 2],
                    [0, 0, 1]],
                    dtype=np.float32
                )
            rotation = decenter @ rotation @ center

            translate = \
                np.array([
                    [1, 0, dx],
                    [0, 1, dy],
                    [0, 0, 1]],
                    dtype=np.float32
                )
            T = translate @ rotation @ flip @ resize
        else:
            T = resize

        image = cv2.warpAffine(image, T[:2, :], self.shape)
        ones = np.ones(len(landmarks), dtype=np.float32)
        x = np.c_[landmarks, ones]
        landmarks = x @ np.transpose(T[:2, :])
        if self.phase == 'train':
            image = utils.random_gamma_trans(image, np.random.uniform(0.8, 1.2, 1))
            image = utils.random_color(image)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        landmarks[:, 0] /= self.shape[1]
        landmarks[:, 1] /= self.shape[0]
        return image, np.reshape(landmarks, (-1))


if __name__ == '__main__':
    a = BBoxDataset('/data/icme/crop/data/picture',
                    '/data/icme/crop/data/landmark',
                    '/data/icme/train')
    import matplotlib.pyplot as plt
    for i in range(100):
        image, landmark = a.__getitem__(i)
        image = np.transpose(image, (1, 2, 0))[:, :, ::-1].astype(np.uint8)
        landmark = np.reshape(landmark, [-1, 2])
        plt.imshow(image)
        plt.scatter(landmark[:, 0], landmark[:, 1])
        plt.show()