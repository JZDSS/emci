import numpy as np
import cv2
import utils

imgdir = '/home/zhzhong/Desktop/test.jpg'
img = cv2.imread(imgdir)
cv2.imshow('test',img)
cv2.waitKey(0)

img = utils.random_flip(img, prob = 0.8)
cv2.imshow('test1',img)
cv2.waitKey(0)

img = utils.random_gamma_trans(img,0.5)
cv2.imshow('test2',img)
cv2.waitKey(0)
