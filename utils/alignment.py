import torch
import numpy as np
from sklearn.externals import joblib
import pdb
import cv2
import data.read_dataset as fd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import data.utils as ul
from PIL import Image

mean_landmarks = joblib.load('../cache/mean_landmarks.pkl')

def inv_norm_landmark(landmark, bbox):
    """
    根据bounding box信息对landmarks坐标进行规范化。
    :param landmarks: 形如(N, 2)的矩阵，其中N为landmark个数
    :param bbox: [minx, miny, maxx, maxy]
    :return: 规范化landmarks，形状同landmarks输入。
    """
    minx, miny, maxx, maxy = bbox
    w = float(maxx - minx)
    h = float(maxy - miny)
    landmark[:, 0] = landmark[:, 0] * w
    landmark[:, 1] = landmark[:, 1] * h
    landmark += [minx, miny]
    return landmark

def align(images, landmarks, bboxs):
    ones = np.ones(106)
    batch_size = int(images.shape[0])

    bboxs_origin = bboxs.data.numpy()

    for i in range(batch_size):
        image = images[i].data.numpy()

        landmark = landmarks[i].data.numpy()
        landmark = landmark.reshape((106, 2), order='C')
        #landmark = np.reshape(landmark, (106, 2))
        bbox = bboxs_origin[i].astype(np.int)
        landmark = inv_norm_landmark(landmark, np.reshape(bbox, (-1))).astype(np.int)
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8).copy()

        plt.figure()
        plt.imshow(image)
        plt.scatter(landmark[:, 0], landmark[:, 1])
        plt.plot(bbox[:, 0], bbox[:, 1])
        plt.show()


        x = np.c_[landmark, ones]
        T = pdb.procrustes(x, mean_landmarks*128)
        landmark = x @ T

        bbox = bbox.reshape((2, 2))
        bbox = np.c_[bbox, [1, 1]]
        bbox = bbox @ T

        cols, rows, ch = image.shape
        image = cv2.warpAffine(image, np.transpose(T), (cols, rows))


        plt.figure()
        plt.imshow(image)
        # plt.show()
        plt.scatter(landmark[:, 0], landmark[:, 1])
        plt.plot(bbox[:, 0], bbox[:, 1])

        plt.xlim(0, 128)
        plt.ylim(128, 0)
        plt.show()


    return images

if __name__ == "__main__":
    a = fd.FaceDataset("../data/icme", "../data/icme/train")
    b = iter(DataLoader(a, batch_size=100, shuffle=True, num_workers=0))
    while True:
        images, landmarks, bboxs = next(b)
        images = align(images, landmarks, bboxs)