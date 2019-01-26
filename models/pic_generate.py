import os
from utils import pwa
from data import utils
import numpy as np
import matplotlib.pyplot as plt

filename = os.listdir("/home/zhzhong/Desktop/correctdata/train/1/")
img_dir = '/home/zhzhong/Desktop/correctdata/train/1/'
base_dir = "/home/zhzhong/Desktop/correctdata/bbox/"
new_dir = "/home/zhzhong/Desktop/newdata/1/"
bbox1 = np.array([0, 0, 128, 128])
bbox2 = np.array([0, 0, 128, 128])
dst = utils.read_mat('/home/zhzhong/Desktop/correctdata/data/landmark/AFW_5452623_1_5.jpg.txt')
dst = utils.norm_landmarks(dst, bbox2)
filenameforlmk = os.listdir("/home/zhzhong/Desktop/correctdata/data/landmark/")
base_diroflmk = "/home/zhzhong/Desktop/correctdata/data/landmark/"
for img in filename:
    image = plt.imread(img_dir + img)
    # plt.imshow(image)
    # plt.show()
    box_dir = "/home/zhzhong/Desktop/correctdata/bbox/" + img + '.rect'
    box = utils.read_bbox(box_dir)
    image = image[box[1]: box[3], box[0]: box[2]]
    for lmk in filenameforlmk:
        src = utils.read_mat(base_diroflmk + lmk)
        src = utils.norm_landmarks(src, bbox1)
        image_pwa,_ = pwa.pwa(image, src, dst, (128, 128))
        #image_pwa.save(new_dir + img)
        plt.imshow(image_pwa)
        plt.show()

