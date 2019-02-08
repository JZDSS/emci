import os
import cv2
import numpy as np
from data.align_dataset import AlignDataset
from data import utils

class AlignFusionDataset(AlignDataset):

    def __init__(self,
                 img_dir,
                 gt_ldmk_dir,
                 al_ldmk_dir,
                 bin_dir,
                 aligner,
                 bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 phase='train',
                 shape=(224, 224),
                 flip=True,
                 ldmk_ids=[i for i in range(106)], kernel_size=15, sigma=5):
        super(AlignFusionDataset, self).__init__(img_dir, gt_ldmk_dir, al_ldmk_dir, bin_dir, aligner,
                                                 bins, phase, shape, flip, ldmk_ids)
        self.aligner = aligner
        self.algin_ldmk = [os.path.join(al_ldmk_dir, f + '.txt') for f in self.file_list]
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __getitem__(self, item):
        image, landmarks = super(AlignFusionDataset, self).__getitem__(item)
        image = np.transpose(image, [1, 2, 0])
        landmarks = np.reshape(landmarks, (-1, 2))

        landmarks[:, 0] *= self.shape[1]
        landmarks[:, 1] *= self.shape[0]
        # heatmap = np.zeros((self.shape[0], self.shape[1], 3), np.uint8)

        kernel_size = self.kernel_size
        sigma = self.sigma
        heatmaps = []
        # 脸部外轮廓
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = list(range(32))
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 鼻梁
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = [51, 52, 53, 60]
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 鼻子下轮廓
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = list(range(55, 66))
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 左眉上
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = list(range(33, 38))
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 左眉下
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = [33, 41, 40, 39, 38, 37]
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 右眉上
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = list(range(42, 47))
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 右眉下
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = [46, 47, 48, 49, 50, 42]
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 左眼上
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = list(range(66, 71))
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 左眼下
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = [66, 73, 72, 71, 70]
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 右眼上
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = list(range(75, 80))
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 右眼下
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = [75, 82, 81, 80, 79]
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 上嘴唇上
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = [84, 85, 86, 87, 88, 89, 90]
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 上嘴唇下
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = [90, 100, 99, 98, 97, 96, 84]
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 下嘴唇上
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = [84, 103, 102, 101, 90]
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))

        # 下嘴唇下
        img = np.zeros((self.shape[0], self.shape[1]), np.uint8)
        idx = [90, 91, 92, 93, 94, 95, 84]
        draw_curve(img, landmarks, idx)
        heatmaps.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))
        inputs = []
        for i in range(len(heatmaps)):
            heatmaps[i] = heatmaps[i].astype(np.float32) / np.max(heatmaps[i])
            inputs.append(image * np.expand_dims(heatmaps[i], -1))

        inputs.append(image)
        inputs = np.concatenate(inputs, axis=-1)
        inputs = np.transpose(inputs, [2, 0, 1])
        heatmaps = np.stack(heatmaps, axis=2)
        heatmaps = np.transpose(heatmaps, [2, 0, 1])
        landmarks[:, 0] /= self.shape[1]
        landmarks[:, 1] /= self.shape[0]
        return inputs, np.reshape(landmarks, (-1)), heatmaps


# def draw_circle(img, landmarks, idx):
#     # idx = list(range(33, 41))
#     idx.append(idx[0])
#     draw_curve(img, landmarks, idx)

def draw_curve(img, landmarks, idx):
    for j in range(len(idx) - 1):
        cv2.line(img, (landmarks[idx[j], 0], landmarks[idx[j], 1]),
                 (landmarks[idx[j + 1], 0], landmarks[idx[j + 1], 1]), (255, 255, 255),
                     thickness=4)
