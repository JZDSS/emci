import numpy as np
from data import utils
import cv2
from utils.pwa import pwa
from PIL import Image
import matplotlib.pyplot as plt
import os
from random import shuffle
import time
from sklearn.externals import joblib

#在/data/icme/下新建expand文件夹用于存储扩充的数据，expand
#下新建11个文件夹，名字1～11,再新建landmarks文件夹用于存储
#变换后的landmark

U = joblib.load('cache/u.pkl')
in_image_dir = '/data/icme/train'
in_landmark_dir = '/data/icme/data/landmark'
in_bbox_dir = '/data/icme/bbox'
out_dir = '/data/icme/expand'

#multiply = 1.01

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
        out, new_landmark = pwa(self.image, src, dst, (128, 128))
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

    def random_distort(self):
        r = np.random.randint(0, 5, 1)
        #r = 3
        if r == 0:
            return self.mouth()
        elif r == 1:
            return self.nose()
        elif r == 2:
            return self.eye_l()
        elif r == 3:
            return self.eye_r()
        elif r == 4:
            return self.brow_l()
        elif r == 5:
            return self.brow_r()


def save_rect(rect, file):
    with open(file, 'w') as f:
            f.write('%d %d %d %d' % rect)


if __name__ == "__main__":
    for i in range(1, 12):
        pose = '%d' % i
        curr_path = os.path.join(in_image_dir, pose)
        image_names = os.listdir(curr_path)
        image_nums = len(image_names)
        #multiply = float(1 / image_nums) + 1
        #N = int(image_nums * (multiply - 1))
        N = 1
        n = 0
        for src_name in image_names:
            print(curr_path, n)
            image = plt.imread(os.path.join(curr_path, src_name))
            # image.flags.writeable = True  # 改图像可读写
            src_landmark = utils.read_mat(os.path.join(in_landmark_dir, src_name + '.txt'))
            src_bbox = utils.read_bbox(os.path.join(in_bbox_dir, src_name + '.rect'))
            src_landmark = utils.norm_landmarks(src_landmark, src_bbox)
            face_image = image[src_bbox[1]: src_bbox[3], src_bbox[0]: src_bbox[2]]
            t = 0
            names = image_names.copy()
            names.remove(src_name)
            shuffle(names)
            dst_names = names[0 : N]
            for dst_name in dst_names:
                temp_image = image.copy()
                temp_image.flags.writeable = True  # 改图像可读写
                temp_face = face_image.copy()

                dst_landmark = utils.read_mat(os.path.join(in_landmark_dir, dst_name + '.txt'))
                dst_bbox = utils.read_bbox(os.path.join(in_bbox_dir, dst_name + '.rect'))
                dst_landmark = utils.norm_landmarks(dst_landmark, dst_bbox)

                #变换面部的部分
                transform = expandpose(temp_face, src_landmark, dst_landmark)
                temp_face2, new_landmark = transform.random_distort()
                # dst_landmark[0: 33, :] = src_landmark[0: 33, :]
                # temp_face2, new_landmark = pwa(temp_face, src_landmark.copy(), dst_landmark.copy(), (128, 128))
                temp_face2 = cv2.resize(temp_face2, (temp_face.shape[1], temp_face.shape[0]))

                # #显示原面部图像
                # plt.imshow(temp_face)
                # plt.show()
                #
                # #显示变换后的面部图像
                # plt.imshow(temp_face2)
                # plt.show()

                #创建新名字
                temp_src = src_name[0 : len(src_name) - 4]
                image_new_name = temp_src + '_e' + str(t) + '.jpg'
                t += 1

                #保存变换后的图像
                temp_image[src_bbox[1]: src_bbox[3], src_bbox[0]: src_bbox[2]] = (temp_face2 * 256).astype(np.uint8)
                image_file = Image.fromarray(temp_image).convert('RGB')
                image_file.save(out_dir + '/' + pose + '/' + image_new_name)

                # 保存landmark
                new_landmark = utils.inv_norm_landmark(new_landmark, src_bbox).astype(int)
                utils.save_landmarks(new_landmark, out_dir + '/' + 'landmarks' + '/' + image_new_name + '.txt')  # 归一化的landmark

                #保存bbox
                # save_rect(src_bbox, out_dir + '/' + 'bbox' + '/' + image_new_name + '.rect')

                #显示变换后的图片与landmark
                #landmark2 = utils.inv_norm_landmark(src_landmark, src_bbox)
                # for j in range(106):
                #     cv2.circle(temp_image, (new_landmark[j, 0], new_landmark[j, 1]), 1, (0, 255, 0))
                # cv2.imshow('test', temp_image)
                # cv2.waitKey(1)
                n += 1