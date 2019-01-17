import numpy as np
import cv2

def read_bbox(path):
    """
    从文件读取bounding box 坐标
    :param path: rect文件路径
    :return: [min_x, min_y, max_x, max_y]
    """
    with open(path) as f:
        l = f.readline().strip('\n').split(' ')
    return [int(ll) for ll in l]


def read_landmarks(path):
    """
    从文件读取landmark坐标
    :param path: txt文件路径
    :return: 形状为(106, 2)的numpy.ndarray
    """
    landmarks = []
    with open(path) as f:
        n = int(f.readline())
        for i in range(n):
            pt = f.readline().strip('\n').split(' ')
            pt = [int(float(p)) for p in pt]
            landmarks.append(pt)
    return np.array(landmarks, dtype=np.float32)

def norm_landmarks(landmarks, bbox):
    """
    根据bounding box信息对landmarks坐标进行规范化。
    :param landmarks: 形如(N, 2)的矩阵，其中N为landmark个数
    :param bbox: [minx, miny, maxx, maxy]
    :return: 规范化landmarks，形状同landmarks输入。
    """
    minx, miny, maxx, maxy = bbox
    w = float(maxx - minx)
    h = float(maxy - miny)
    landmarks -= [minx, miny]
    landmarks[:, 0] = landmarks[:, 0] / w
    landmarks[:, 1] = landmarks[:, 1] / h
    return landmarks

def draw_landmarks(image, landmarks, color):
    landmarks = np.reshape(landmarks, (106, 2))
    image = np.transpose(image, (1, 2, 0)).astype(np.uint8).copy()
    landmarks[:, 0] *= image.shape[1]
    landmarks[:, 1] *= image.shape[0]
    landmarks.astype(np.int)
    for j in range(106):
        cv2.circle(image, (landmarks[j, 0], landmarks[j, 1]), 2, color)
    return np.transpose(image, (2, 0, 1))
