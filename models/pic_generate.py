import os
from utils import pwa
from data import utils
import numpy as np
import matplotlib.pyplot as plt

filename = os.listdir("/home/zhang/PWA-AL/1/")
base_dir = "/home/zhang/PWA-AL/1/"
new_dir = "/home/zhang/PWA-AL/2/"
bbox1 = np.array([0, 0, 128, 128])
bbox2 = np.array([0, 0, 128, 128])
dst = utils.read_landmarks('/home/zhang/aligned/landmark/AFW_261068_1_1.jpg.txt')
dst = utils.norm_landmarks(dst, bbox2)
filenameforlmk = os.listdir("/home/zhang/PWA-AL/1lmk/")
base_diroflmk = "/home/zhang/PWA-AL/1lmk/"
for img in filename:
    image = plt.imread(base_dir + img)
    image = image[bbox1[1]: bbox1[3], bbox1[0]: bbox1[2]]
    for lmk in filenameforlmk:

        if img == lmk.strip('.txt'):
            src = utils.read_landmarks(base_diroflmk + lmk)
            src = utils.norm_landmarks(src, bbox1)
            image_pwa = pwa.pwa(image, src, dst, (128, 128))
            #image_pwa.save(new_dir + img)
            plt.imshow(image_pwa)
            plt.show()

