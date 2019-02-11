import numpy as np
import cv2
from PIL import Image, ImageEnhance


def read_bbox(path):
    """
    从文件读取bounding box 坐标
    :param path: rect文件路径
    :return: [min_x, min_y, max_x, max_y]
    """
    with open(path) as f:
        l = f.readline().strip('\n').split(' ')
    return [int(ll) for ll in l]


def read_mat(path):
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
            pt = [float(p) for p in pt]
            landmarks.append(pt)
    return np.array(landmarks, dtype=np.float32)


def save_landmarks(landmarks, file):
    """
    保存landmarks
    :param landmarks: (106, 2)
    :param file: 文件
    """
    with open(file, 'w') as f:
        f.write("%d\n" % 106)
        for i in range(106):
            f.write('%f %f\n' % (landmarks[i, 0], landmarks[i, 1]))


def save_T(T, file):
    with open(file, 'w') as f:
        f.write("%d\n" % 3)
        for i in range(3):
            f.write('%f %f\n' % (T[i, 0], T[i, 1]))


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


def inv_norm_landmark(landmark, bbox):
    """
    注释有误
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


def draw_landmarks(image, landmarks, color):
    landmarks = np.reshape(landmarks, (-1, 2))
    image = np.transpose(image, (1, 2, 0)).astype(np.uint8).copy()
    landmarks[:, 0] *= image.shape[1]
    landmarks[:, 1] *= image.shape[0]
    landmarks.astype(np.int)
    for j in range(landmarks.shape[0]):
        cv2.circle(image, (landmarks[j, 0], landmarks[j, 1]), 2, color)
    return np.transpose(image, (2, 0, 1))


def random_flip(img, landmark, prob):
    a = np.random.uniform(0,1,1)
    if a < prob:
        img = cv2.flip(img,1)
        landmark = landmark_flip(landmark)
    return img, landmark


def landmark_flip(landmark, max_x=1):
    landmark[:, 0] = max_x - landmark[:, 0]
    idx = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,
           3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49,
           48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52, 53, 54, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 79,
           78, 77, 76, 75, 82, 81, 80, 83, 70, 69,
           68, 67, 66, 73, 72, 71, 74, 90, 89, 88, 87, 86, 85, 84, 95, 94, 93, 92, 91, 100, 99, 98, 97, 96, 103, 102,
           101, 105, 104]
    landmark = landmark[idx]
    return landmark


def random_gamma_trans(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)


def random_color(img):
    img = Image.fromarray(img)

    random_factor = np.random.uniform(0.5, 1.5)
    color_img = ImageEnhance.Color(img).enhance(random_factor)

    random_factor = np.random.uniform(0.5, 1.5)
    brightness_img = ImageEnhance.Brightness(color_img).enhance(random_factor)

    random_factor = np.random.uniform(0.5, 1.5)
    contrast_img = ImageEnhance.Contrast(brightness_img).enhance(random_factor)

    random_factor = np.random.uniform(0.5, 1.5)
    sharpness_img = ImageEnhance.Sharpness(contrast_img).enhance(random_factor)
    return np.array(sharpness_img)