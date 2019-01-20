import numpy as np
from sklearn.externals import joblib
import pdb
import cv2


class Align(object):

    def __init__(self, reference_path='../cache/mean_landmarks.pkl', scale=(128, 128), margin=(0.15, 0.1)):
        """
        :param reference_path: 参考landmark的路径
        :param scale: 输出图片大小，tuple
        :param margin: tuple，(x_margin, y_margin)人脸和边界之间的距离，左右margin为x_margin*W,上下margin为y_margin*H，
                       其中W和H为人脸的宽度和高度
        """
        self.reference = joblib.load(reference_path)
        # plt.subplot(1, 2, 1)
        # plt.scatter(self.reference[:, 0], self.reference[:, 1])
        max_y, min_y, max_x, min_x = [max(self.reference[:, 1]), min(self.reference[:, 1]),
                                      max(self.reference[:, 0]), min(self.reference[:, 0])]
        h = max_y - min_y
        w = max_x - min_x
        margin_x, margin_y = margin
        k_x, b_x = [1 / w, margin_x - min_x / w]
        k_y, b_y = [1 / h, margin_y - min_y / h]

        self.reference[:, 0] = (k_x * self.reference[:, 0] + b_x) / (2 * margin_x + 1) * scale[0]
        self.reference[:, 1] = (k_y * self.reference[:, 1] + b_y) / (2 * margin_y + 1) * scale[1]
        self.scale = scale
        # plt.subplot(1, 2, 2)
        # plt.scatter(self.reference[:, 0], self.reference[:, 1])
        # plt.xlim(0, scale[0])
        # plt.ylim(0, scale[1])
        # plt.show()

    def __call__(self, image, landmarks, bbox):
        """
        :param image: (H, W, 3)
        :param landmarks: (N, 2), unnormalized
        :param bbox: [[min_x, min_y]
                       max_x, max_y]]
        :return: aligned image
        """
        ones = np.ones(106)
        x = np.c_[landmarks, ones]

        T = pdb.procrustes(x, self.reference)
        landmarks = x @ T

        image = cv2.warpAffine(image, np.transpose(T), self.scale)
        return image, landmarks

# 示例代码：
# import os
# root_dir = '/data/icme'
# bin_dir = '/data/icme/train'
# pose = 1
# a = Align()
# bins = os.listdir(bin_dir)
#
# file_list = []
# b = bins[pose]
# curr = os.path.join(bin_dir, b)
# files = os.listdir(curr)
# for i in files:
#     file_list.append(i)
# for i in range(100):
#     img_dir = os.path.join(root_dir, 'data/picture')
#     landmark_dir = os.path.join(root_dir, 'data/landmark')
#     bbox_dir = os.path.join(root_dir, 'bbox')
#     images = [os.path.join(img_dir, f) for f in file_list]
#     landmarks = [os.path.join(landmark_dir, f + '.txt') for f in file_list]
#     bboxes = [os.path.join(bbox_dir, f + '.rect') for f in file_list]
#     img_path = images[i]
#     bbox_path = bboxes[i]
#     landmark_path = landmarks[i]
#     bbox = ul.read_bbox(bbox_path)
#     landmarks = ul.read_landmarks(landmark_path)
#     image = cv2.imread(img_path)
#
#     image, landmark = a(image, landmarks, bbox)
#
#     plt.imshow(image)
#     plt.scatter(landmark[:, 0], landmark[:, 1])
#     # plt.scatter(self.reference[:, 0], self.reference[:, 1])
#     # plt.plot(bbox[:, 0], bbox[:, 1])
#     plt.xlim(0, a.scale[0])
#     plt.ylim(a.scale[1], 0)
#     plt.show()
