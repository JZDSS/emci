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
from utils import alignment
import pdb

#在/data/icme/下新建expand文件夹用于存储扩充的数据，expand
#下新建11个文件夹，名字1～11,再新建landmarks文件夹用于存储
#变换后的landmark


in_image_dir = '/data/icme/train'
in_landmark_dir = '/data/icme/data/landmark'
in_bbox_dir = '/data/icme/bbox'
out_dir = '/data/icme/expand'
aligner = alignment.Align('cache/mean_landmarks.pkl')
mean_landmark = joblib.load('cache/mean_landmarks.pkl')
M = 1
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
        out, _ = pwa(self.image, src.copy(), dst.copy(), (128, 128))
        return out, dst

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
        r = np.random.randint(0, 6, 1)
        #r = 3
        if r == 0:
            s = 'mouth'
            return self.mouth(), s
        elif r == 1:
            s = 'nose'
            return self.nose(), s
        elif r == 2:
            s = 'eye_l'
            return self.eye_l(), s
        elif r == 3:
            s = 'eye_r'
            return self.eye_r(), s
        elif r == 4:
            s = 'brow_l'
            return self.brow_l(), s
        elif r == 5:
            s = 'brow_r'
            return self.brow_r(), s

def get_iamge_index(pose):
    aligned = joblib.load(out_dir + '/cache/' + pose + '/aligned.pkl')
    temp = (aligned - np.mean(aligned, axis=0))
    U = joblib.load(out_dir + '/cache/' + pose + '/u.pkl')
    pc = temp.dot(U[:, 0])
    idx = np.argsort(pc)
    return idx

def normal(landmark):
    minx, miny = np.min(landmark, axis=0)
    maxx, maxy = np.max(landmark, axis=0)
    landmark -= [minx, miny]
    landmark[:, 0] /= (maxx - minx)
    landmark[:, 1] /= (maxy - miny)
    return landmark

def Alignment(src_image, src_landmark, dst_landmark):
    """
    :param src_image: 源图像
    :param src_landmark: 归一化源landmark
    :param dst_landmark: 归一化目标landmark
    :return: 对齐的图像
    """
    ones = np.ones(106, dtype=np.float32)
    x = np.c_[src_landmark, ones]
    T = pdb.procrustes(x, dst_landmark)
    T[2, :] = 0
    landmark = x @ T
    rows = src_image.shape[1]
    cols = src_image.shape[0]
    src_image = np.pad(src_image, ((rows//2, rows//2), (cols//3, cols//3), (0, 0)), 'constant')
    image = cv2.warpAffine(src_image, np.transpose(T), (src_image.shape[1], src_image.shape[0]))
    return image, landmark, T

def inverse_image(image, T, ):
    inv_T = np.zeros((3, 2))
    #inv_T[2, :] =  -T[2, :]
    inv_T[0:2, :] = np.linalg.inv(T[0:2, :])
    print(inv_T)
    inv_image = cv2.warpAffine(image, np.transpose(inv_T), (256, 256))
    return inv_image

if __name__ == "__main__":

    for i in range(1, 12):
        pose = '%d' % i
        curr_path = os.path.join(in_image_dir, pose)
        filenames = np.array(joblib.load(out_dir + '/cache/' + pose + '/filenames.pkl'))
        idx = get_iamge_index(pose)
        n = 0
        for j in range(0, len(idx) - M):
            print(curr_path, len(filenames), n)
            id = list(idx[j: j + M + 1])
            filename = filenames[id]
            src_name = filename[0]
            image = plt.imread(os.path.join(curr_path, src_name))
            # image.flags.writeable = True  # 改图像可读写
            src_landmark = utils.read_mat(os.path.join(in_landmark_dir, src_name + '.txt'))
            src_bbox = utils.read_bbox(os.path.join(in_bbox_dir, src_name + '.rect'))
            face_image = image[src_bbox[1]: src_bbox[3], src_bbox[0]: src_bbox[2]]
            #src_landmark = utils.norm_landmarks(src_landmark, src_bbox)
            #temp_image = np.pad(face_image, ((10, 10),(10, 10),(0, 0)), 'constant')
            src_landmark = normal(src_landmark)
            mean_landmark = normal(mean_landmark)
            ali_face_image, ali_src_landmark, T = Alignment(face_image.copy(), src_landmark, mean_landmark)
            plt.subplot(1,3,1)
            plt.imshow(face_image)
            plt.subplot(1,3,2)
            plt.imshow(face_image)
            plt.subplot(1,3,3)
            plt.imshow(ali_face_image)
            plt.show()
            for k in range(1, M + 1):
                dst_name = filename[k]

                dst_landmark = utils.read_mat(os.path.join(in_landmark_dir, dst_name + '.txt'))

                _, ali_dst_landmark, _ = aligner(image.copy(), dst_landmark.copy())
                ali_dst_landmark[:, 0] /= aligner.scale[0]
                ali_dst_landmark[:, 1] /= aligner.scale[1]
                # dst_bbox = utils.read_bbox(os.path.join(in_bbox_dir, dst_name + '.rect'))
                # dst_landmark = utils.norm_landmarks(dst_landmark, dst_bbox)

                #变换面部的部分
                transform = expandpose(ali_image.copy(), ali_src_lanmark.copy(), ali_dst_landmark.copy())
                distort_ali_image, new_landmark = transform.random_distort()
                # ali_dst_landmark[0: 33, :] = ali_src_lanmark[0: 33, :]
                # distort_image, new_landmark = pwa(ali_image.copy(), ali_src_lanmark.copy(), ali_dst_landmark.copy(), (128, 128))
                # face_image2 = cv2.resize(face_image2, (face_image.shape[1], face_image.shape[0]))

                # print(T.shape)
                # distort_image = aligner.inverse_image(distort_ali_image, T)

                #显示原面部图像
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                #plt.show()

                #显示变换后的面部图像
                plt.subplot(1, 3, 2)
                plt.imshow(ali_image)
                #plt.show()

                plt.subplot(1, 3, 3)
                plt.imshow(distort_ali_image)
                plt.show()

                # plt.imshow(distort_image)
                # plt.show()

                # #创建新名字
                # temp_src = src_name[0 : len(src_name) - 4]
                # image_new_name = temp_src + '_e' + str(k - 1) + '.jpg'
                #
                # #保存变换后的图像
                # image_file = Image.fromarray(distort_image).convert('RGB')
                # image_file.save(out_dir + '/' + pose + '/' + image_new_name)
                #
                # # 保存landmark
                # new_landmark = utils.inv_norm_landmark(new_landmark, src_bbox).astype(int)
                # utils.save_landmarks(new_landmark, out_dir + '/' + 'landmarks' + '/' + image_new_name + '.txt')  # 归一化的landmark
                #
                # #显示变换后的图片与landmark
                # #landmark2 = utils.inv_norm_landmark(src_landmark, src_bbox)
                # for j in range(106):
                #     cv2.circle(image2, (new_landmark[j, 0], new_landmark[j, 1]), 1, (0, 255, 0))
                # cv2.imshow('test', image2)
                # cv2.waitKey(0)
                time.sleep(3)
                n += 1