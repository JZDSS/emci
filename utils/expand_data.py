import numpy as np
from data import utils
import cv2
from utils.pwa import pwa

class expandpose():
    def __init__(self, image, src_landmark, dst_landmark):
        """
        :param image: 输入图像
        :param src_landmark: 原图像归一化landmark
        :param dst_landmark: 目标图像归一化landmark
        """
        self.image = image
        self.src_landmark = src_landmark
        self.dst_landmark = dst_landmark
        self.mouth_range = np.arange(84, 104)
        self.nose_range = np.arange(52, 65)
        self.eyer_range = np.arange(66, 75)
        self.eyel_range = np.arange(75, 84)
        self.browr_range = np.arange(33, 42)
        self.browl_range = np.arange(42, 51)

    def transform(self, x):
        src = self.src_landmark.copy()
        dst = self.src_landmark.copy()
        dst[x, :] = self.dst_landmark[x, :]
        out, new_landmark = pwa(self.image, src, dst, (self.image.shape[1], self.image.shape[0]))
        return out, new_landmark

    def mouth(self):
        x = self.mouth_range
        return self.transform(x)

    def nose(self):
        x = self.nose_range
        return  self.transform(x)

    def eye_l(self):
        x = self.eyel_range
        return self.transform(x)

    def eye_r(self):
        x = self.eyer_range
        return self.transform(x)

    def brow_l(self):
        x = self.browl_range
        return self.transform(x)

    def brow_r(self):
        x = self.browr_range
        return self.transform(x)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = plt.imread('/data/icme/data/picture/AFW_134212_1_0.jpg', )
    image.flags.writeable = True#改图像可读写

    src_landmark = utils.read_mat('/data/icme/data/landmark/AFW_134212_1_0.jpg.txt')
    src_bbox = utils.read_bbox('/data/icme/bbox/AFW_134212_1_0.jpg.rect')
    src_landmark = utils.norm_landmarks(src_landmark, src_bbox)

    face_image = image[src_bbox[1]: src_bbox[3], src_bbox[0]: src_bbox[2]]

    dst_landmark = utils.read_mat('/data/icme/data/landmark/AFW_134212_1_3.jpg.txt')
    dst_bbox = utils.read_bbox('/data/icme/bbox/AFW_134212_1_3.jpg.rect')
    dst_landmark = utils.norm_landmarks(dst_landmark, dst_bbox)

    #变换面部的部分
    transform = expandpose(face_image, src_landmark, dst_landmark)
    face_image2, landmark = transform.nose()

    #显示原面部图像
    plt.imshow(face_image)
    plt.show()

    #显示变换后的面部图像
    plt.imshow(face_image2)
    plt.show()

    #图片与landmark的路径
    PATH = "/data/icme/expand/1/"
    image_name = "AFW_134212_1_4.jpg"
    txt_name = "AFW_134212_1_4.txt"

    #保存landmark
    utils.save_landmarks(landmark, PATH + txt_name)#归一化的landmark

    #保存变换后的图像
    image[src_bbox[1]: src_bbox[3], src_bbox[0]: src_bbox[2]] = (face_image2 * 256).astype(np.uint8)
    plt.imshow(image)
    plt.savefig(PATH + image_name)

    #显示变换后的图片与landmark
    landmark = utils.inv_norm_landmark(landmark, src_bbox).astype(int)
    #landmark2 = utils.inv_norm_landmark(src_landmark, src_bbox)
    for j in range(106):
        cv2.circle(image, (landmark[j, 0], landmark[j, 1]), 2, (0, 255, 0))
    cv2.imshow('test', image)
    cv2.waitKey(0)